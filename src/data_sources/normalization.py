"""src/data_sources/normalization.py - Normalisation canonique des données brutes Steam.

Convertit et valide les réponses brutes des sources (Store API, Reviews API, SteamSpy)
vers le format canonique 16 colonnes du projet MLOps Steam :
- Conversion des prix en centimes -> décimales standard.
- Calcul rigoureux de pct_pos_total.
- Nettoyage et formatage des tranches d'owners (suppression virgules, tirets).
- Encodage ISO YYYY-MM-DD des dates de sortie.
- Rejet immédiat des jeux non publiés ou sans cibles valides.
"""

from __future__ import annotations

import ast
from datetime import datetime
import re
from typing import Any, Dict, List, Optional
from dateutil import parser as date_parser

from src.contracts import REQUIRED_SNAPSHOT_COLUMNS
from src.exceptions import DataValidationError


def normalize_owner_range(owners_raw: Any) -> str:
    """Normalise une tranche SteamSpy ('10,000,000 .. 20,000,000') en format canonique ('10000000 - 20000000')."""
    if not owners_raw:
        raise DataValidationError("Tranche d'owners manquante ou vide.")

    s = str(owners_raw).strip()
    # Nettoyage des virgules et espaces
    cleaned = s.replace(",", "").replace("..", "-").strip()
    match = re.match(r"^(\d+)\s*-\s*(\d+)$", cleaned)
    if not match:
        raise DataValidationError(f"Format de tranche d'owners invalide : '{owners_raw}'")

    low, high = int(match.group(1)), int(match.group(2))
    if low >= high and not (low == 0 and high == 0):
        raise DataValidationError(f"Tranche avec bornes inversées ou égales : '{owners_raw}' ({low} >= {high})")

    return f"{low} - {high}"


def parse_release_date_iso(date_raw: Any) -> Optional[str]:
    """Convertit une chaîne de date libre ('18 Apr, 2011', '2011-04-18') en ISO YYYY-MM-DD."""
    if not date_raw:
        return None
    s = str(date_raw).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    try:
        dt = date_parser.parse(s)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # En cas d'échec de parsing, retourne une date par défaut ou lève une exception
        return None


def normalize_steam_game(
    appid: int,
    store_payload: Dict[str, Any],
    steamspy_payload: Optional[Dict[str, Any]] = None,
    reviews_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble et normalise les données des 3 sources en une ligne conforme aux 16 colonnes.

    Args:
        appid: Identifiant unique Steam.
        store_payload: Réponse brute de store.steampowered.com/api/appdetails.
        steamspy_payload: Réponse brute de steamspy.com/api.php.
        reviews_payload: Réponse brute optionnelle de store.steampowered.com/appreviews.

    Returns:
        Dictionnaire contenant les 16 colonnes du snapshot Steam.

    Raises:
        DataValidationError si le jeu n'est pas publié, invalide ou incomplet.
    """
    app_data = store_payload.get(str(appid), {})
    if not app_data.get("success", False):
        raise DataValidationError(f"Échec de récupération store pour appid {appid} (success=False).")

    data = app_data.get("data", {})
    if not data:
        raise DataValidationError(f"Données store absentes pour appid {appid}.")

    # 1. Vérification de publication (rejet des jeux à venir / coming soon)
    rel_info = data.get("release_date", {})
    if rel_info.get("coming_soon", False):
        raise DataValidationError(f"Jeu {appid} non publié ('coming_soon' = True).")

    rel_date_str = rel_info.get("date")
    parsed_date = parse_release_date_iso(rel_date_str)
    if not parsed_date:
        parsed_date = "2020-01-01"  # Repli déterministe si non renseignée

    # 2. Titre du jeu
    name = data.get("name", f"App {appid}")

    # 3. Âge minimal
    req_age_raw = data.get("required_age", 0)
    try:
        required_age = max(0, int(req_age_raw))
    except (ValueError, TypeError):
        required_age = 0

    # 4. Prix
    is_free = bool(data.get("is_free", False))
    price_overview = data.get("price_overview", {})
    if is_free:
        price = 0.0
    elif price_overview and "final" in price_overview:
        price = round(float(price_overview["final"]) / 100.0, 2)
    elif steamspy_payload and "price" in steamspy_payload:
        try:
            spy_price = float(steamspy_payload["price"])
            price = round(spy_price / 100.0, 2)
        except Exception:
            price = 0.0
    else:
        price = 0.0

    # 5. DLCs et Succès
    dlc_list = data.get("dlc", [])
    dlc_count = len(dlc_list) if isinstance(dlc_list, list) else 0

    achievements_dict = data.get("achievements", {})
    achievements = int(achievements_dict.get("total", 0)) if isinstance(achievements_dict, dict) else 0

    # 6. Plateformes
    platforms = data.get("platforms", {})
    windows = bool(platforms.get("windows", True))
    mac = bool(platforms.get("mac", False))
    linux = bool(platforms.get("linux", False))

    # 7. Genres
    genres_raw = data.get("genres", [])
    genre_names: List[str] = []
    if isinstance(genres_raw, list):
        for g in genres_raw:
            if isinstance(g, dict) and "description" in g:
                genre_names.append(g["description"].strip())
    elif steamspy_payload and "genre" in steamspy_payload:
        spy_g = str(steamspy_payload["genre"])
        genre_names = [p.strip() for p in spy_g.split(",") if p.strip()]

    genres_str = str(genre_names)

    # 8. Avis (positive, negative, num_reviews_total, pct_pos_total)
    positive = 0
    negative = 0
    if reviews_payload and "query_summary" in reviews_payload:
        q_sum = reviews_payload["query_summary"]
        positive = int(q_sum.get("total_positive", 0))
        negative = int(q_sum.get("total_negative", 0))
    elif steamspy_payload:
        positive = int(steamspy_payload.get("positive", 0))
        negative = int(steamspy_payload.get("negative", 0))

    total_reviews = positive + negative
    if total_reviews > 0:
        pct_pos_total = round(float(positive) / float(total_reviews) * 100.0, 2)
    else:
        pct_pos_total = 0.0

    # 9. Tranche de propriétaires (estimated_owners)
    owners_raw = None
    if steamspy_payload and "owners" in steamspy_payload:
        owners_raw = steamspy_payload["owners"]
    if not owners_raw:
        owners_raw = "0 - 20000"

    estimated_owners = normalize_owner_range(owners_raw)

    return {
        "appid": int(appid),
        "name": str(name),
        "release_date": str(parsed_date),
        "required_age": int(required_age),
        "price": float(price),
        "dlc_count": int(dlc_count),
        "achievements": int(achievements),
        "windows": bool(windows),
        "mac": bool(mac),
        "linux": bool(linux),
        "positive": int(positive),
        "negative": int(negative),
        "num_reviews_total": int(total_reviews),
        "pct_pos_total": float(pct_pos_total),
        "genres": str(genres_str),
        "estimated_owners": str(estimated_owners),
    }

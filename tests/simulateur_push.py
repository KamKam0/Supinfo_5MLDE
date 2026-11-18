"""tests/simulateur_push.py - Simulateur d'envoi d'observations labellisées pour la collecte passive.

Usage :
    uv run --locked python tests/simulateur_push.py 499
    uv run --locked python tests/simulateur_push.py 1
    uv run --locked python tests/simulateur_push.py 1
    uv run --locked python tests/simulateur_push.py 499
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
import time
from typing import Any, Dict, List
from uuid import uuid4
import httpx


OWNER_CLASSES = [
    "20000 - 50000",
    "50000 - 100000",
    "100000 - 200000",
    "200000 - 500000",
    "500000 - 1000000",
]


def generate_synthetic_push_event(seq_id: int, base_appid: int = 2000000) -> Dict[str, Any]:
    """Génère un événement PUSH conforme aux contrats Pydantic et aux attentes Great Expectations."""
    appid = base_appid + seq_id
    pos = 100 + (seq_id % 50) * 10
    neg = 10 + (seq_id % 20)
    total_rev = pos + neg
    pct_pos = round(100.0 * pos / float(total_rev), 2)
    owners = OWNER_CLASSES[seq_id % len(OWNER_CLASSES)]

    return {
        "event_id": f"evt-{appid}-{uuid4().hex[:6]}",
        "source": "simulator",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "game": {
            "appid": appid,
            "name": f"Simulated Push Game {appid}",
            "release_date": "2024-03-15",
            "required_age": 0 if seq_id % 2 == 0 else 18,
            "price": round(float((seq_id % 12) * 2.99), 2),
            "dlc_count": seq_id % 4,
            "windows": True,
            "mac": seq_id % 3 == 0,
            "linux": False,
            "achievements": (seq_id % 10) * 5,
            "genres": "['Action', 'Indie']" if seq_id % 2 == 0 else "['Strategy']",
            "positive": pos,
            "negative": neg,
            "num_reviews_total": total_rev,
            "pct_pos_total": pct_pos,
            "estimated_owners": owners,
        },
    }


def send_batch_events(
    count: int,
    base_url: str = "http://127.0.0.1:8000",
    start_id: Optional[int] = None,
    pause_sec: float = 0.005,
) -> None:
    """Envoie une séquence d'événements PUSH à l'API."""
    games_url = f"{base_url.rstrip('/')}/games"
    status_url = f"{base_url.rstrip('/')}/status"

    counts = {200: 0, 202: 0, 409: 0, 422: 0, "other": 0}
    created_batches: List[str] = []

    t0 = time.time()
    with httpx.Client(timeout=15.0) as client:
        # Contrôle préalable de disponibilité de l'API
        try:
            r = client.get(f"{base_url.rstrip('/')}/health")
            if r.status_code != 200:
                print(f"[ERREUR] Le serveur {base_url} a retourné le code HTTP {r.status_code}")
                sys.exit(1)
        except Exception as e:
            print(f"[ERREUR] Impossible de joindre l'API passive à {base_url} : {e}")
            print("Veuillez vous assurer que le serveur uvicorn est bien démarré sur le port 8000.")
            sys.exit(1)

        # Détection automatique du décalage initial si non spécifié
        if start_id is None:
            try:
                status_resp = client.get(status_url)
                if status_resp.status_code == 200:
                    start_id = int(status_resp.json().get("events_count", 0))
                else:
                    start_id = 0
            except Exception:
                start_id = 0

        print(f"[SIMULATEUR] Début d'envoi de {count} événements vers {games_url} (offset start_id={start_id})...")

        for i in range(count):
            seq_id = start_id + i + 1
            event = generate_synthetic_push_event(seq_id)
            try:
                resp = client.post(games_url, json=event)
                code = resp.status_code
                if code in counts:
                    counts[code] += 1
                else:
                    counts["other"] += 1

                if code == 202:
                    data = resp.json()
                    if data.get("batch_created"):
                        batch_id = data.get("batch_id")
                        created_batches.append(batch_id)
                        print(f"  >>> [LOT CRÉÉ] Déclenchement au seuil de 500 jeux ! ID: {batch_id}")
                elif code != 200:
                    print(f"  [ATTENTION] Réponse HTTP {code} pour l'event {event['event_id']} : {resp.text}")

            except Exception as e:
                print(f"  [ERREUR] Exception réseau : {e}")
                counts["other"] += 1

            if pause_sec > 0:
                time.sleep(pause_sec)

        duration = time.time() - t0
        print(f"[SIMULATEUR] Fin d'envoi en {duration:.2f}s ({count / max(duration, 0.001):.1f} req/s).")
        print(f"  Codes HTTP : 202 (Accepté)={counts[202]}, 200 (Idempotent)={counts[200]}, 409 (Conflit)={counts[409]}, 422={counts[422]}")

        # Consultation de l'état du collecteur
        try:
            status_resp = client.get(status_url)
            if status_resp.status_code == 200:
                s = status_resp.json()
                print("\n[STATUT COLLECTEUR POST-ENVOI] :")
                print(f"  Événements enregistrés        : {s.get('events_count')}")
                print(f"  Jeux uniques non affectés     : {s.get('unassigned_games_count')} (Seuil = {s.get('threshold')})")
                print(f"  Nombre total de lots créés    : {s.get('batches_count')}")
                if created_batches:
                    print(f"  Nouveaux lots constitués      : {created_batches}")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulateur de flux PUSH pour le service de collecte passive")
    parser.add_argument("count", type=int, help="Nombre d'observations à envoyer")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="URL de base du service")
    parser.add_argument("--start-id", type=int, default=None, help="Identifiant de départ (None = détection auto)")
    parser.add_argument("--pause", type=float, default=0.005, help="Pause entre requêtes en secondes")

    args = parser.parse_args()
    send_batch_events(
        count=args.count,
        base_url=args.url,
        start_id=args.start_id,
        pause_sec=args.pause,
    )


if __name__ == "__main__":
    main()

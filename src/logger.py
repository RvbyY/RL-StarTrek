import csv
import time
from pathlib import Path
from typing import Optional
import numpy as np


class EpisodeLogger:

    FIELDNAMES = ["episode", "score", "length", "terminated", "truncated", "reason", "timestamp"]

    def __init__(self, csv_path: str, run_name: Optional[str] = None, verbose: bool = True):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or self.csv_path.stem
        self.verbose = verbose
        self._episodes: list[dict] = []
        self._extra_fields: list[str] = []
        self._file_initialized = False

    def log_episode(self, score, length, terminated, truncated, reason, extra=None):
        extra = extra or {}
        episode_id = len(self._episodes)
        row = {
            "episode":    episode_id,
            "score":      round(float(score), 4),
            "length":     int(length),
            "terminated": terminated,
            "truncated":  truncated,
            "reason":     reason,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"), 
            **extra,
        }
        for k in extra:
            if k not in self._extra_fields:
                self._extra_fields.append(k)
        self._episodes.append(row)
        self._write_row(row)
        if self.verbose:
            self._print_episode(row)

    def print_summary(self, last_n=100):
        if not self._episodes:
            print(f"[{self.run_name}] Aucun épisode logué.")
            return
        recent  = self._episodes[-last_n:]
        scores  = [e["score"]  for e in recent]
        lengths = [e["length"] for e in recent]
        reasons = {}
        for e in recent:
            reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
        print(f"\n{'─'*55}")
        print(f"  {self.run_name}  |  {len(recent)} derniers épisodes")
        print(f"{'─'*55}")
        print(f"  Score    moy={np.mean(scores):+.1f}  std={np.std(scores):.1f}  min={np.min(scores):.1f}  max={np.max(scores):.1f}")
        print(f"  Longueur moy={np.mean(lengths):.0f}")
        print(f"  Raisons  {reasons}")
        print(f"{'─'*55}\n")

    def get_scores(self):
        return [e["score"] for e in self._episodes]

    def get_recent_mean(self, n=100):
        scores = self.get_scores()
        return float(np.mean(scores[-n:])) if scores else float("-inf")

    def is_solved(self, threshold=200.0, window=100):
        return len(self._episodes) >= window and self.get_recent_mean(window) >= threshold

    def num_episodes(self):
        return len(self._episodes)
    def _fieldnames(self):
        return self.FIELDNAMES + self._extra_fields

    def _write_row(self, row):
        write_header = not self._file_initialized
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames(), extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._file_initialized = True
            writer.writerow(row)

    def _print_episode(self, row):
        icon = {"landing": "V", "crash": "X", "out_of_view": "/", "sleep": "Zhhhh"}.get(row["reason"], "?")
        print(
            f"[{self.run_name}] ep={row['episode']:>5}  "
            f"score={row['score']:>+8.1f}  len={row['length']:>4}  "
            f"{icon} {row['reason']:<12}  "
            f"moy100={self.get_recent_mean(100):>+7.1f}"
        )
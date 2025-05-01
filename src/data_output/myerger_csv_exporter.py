from typing import List
from datetime import datetime
import csv
from pathlib import Path


class MyergerCsvExporter:
    def __init__(self, qwk: float = None):
        self.field_names = [
            "_id",
            "true_score",
            "weight",
            "ai_score",
            "cosine_similarity_score",
            "quadratic_weighted_kappa"]
        self.quadratic_weighted_kappa = qwk

    def set_qwk(self, qwk: float) -> None:
        self.quadratic_weighted_kappa = qwk

    def export(self, sample_set_name: str, samples: List[dict]) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = Path(__file__).parent.parent.parent / "data" / f"{sample_set_name}_{timestamp}.csv"
        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.field_names)
            writer.writeheader()
            for rec in samples:
                _id = rec["_id"]
                weight = rec["weight"]
                true_score = rec["normalized_grade"]
                ai_score = rec["ai_response"]["score"]
                cosine_similarity_score = rec["cosine_similarity"]
                writer.writerow({
                    "_id": _id,
                    "true_score": true_score,
                    "weight": weight,
                    "ai_score": ai_score,
                    "cosine_similarity_score": cosine_similarity_score,
                    "quadratic_weighted_kappa": self.quadratic_weighted_kappa
                })

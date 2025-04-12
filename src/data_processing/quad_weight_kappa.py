from typing import List

from sklearn.metrics import cohen_kappa_score

from src.data_processing.metric_generator import MetricGenerator

class QuadWeightKappa(MetricGenerator):

    def generate_metric(self, records: List[dict]) ->  float:
        return self.run_qwk(records)

    def compute_quadratic_weighted_kappa(self, human_scores_list: List[float], predictor_score_list: List[float]) -> float:
        discrete_human_scores = [round(score * 10) for score in human_scores_list]
        discrete_predictor_scores = [round(score * 10) for score in predictor_score_list]
        predictor_qwk = cohen_kappa_score(discrete_human_scores, discrete_predictor_scores, weights="quadratic")
        return predictor_qwk

    def run_qwk(self, sample_list: list[dict]) -> float:
        human_scores = [sample["normalized_grade"] for sample in sample_list]
        ai_scores = [sample["ai_response"]["score"] for sample in sample_list]
        qwk = self.compute_quadratic_weighted_kappa(human_scores, ai_scores)
        return qwk
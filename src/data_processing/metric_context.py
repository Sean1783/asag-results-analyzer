from typing import List

from torch import Tensor

from src.data_processing.metric_generator import MetricGenerator

class MetricContext:
    def __init__(self, metric_generator : MetricGenerator):
        self.metric_generator = metric_generator

    def set_metric_generator(self, metric_generator : MetricGenerator):
        self.metric_generator = metric_generator

    def generate_metric(self, records: List[dict])  -> float | Tensor:
        return self.metric_generator.generate_metric(records)

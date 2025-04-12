from abc import ABC, abstractmethod
from typing import List

from torch import Tensor

class MetricGenerator(ABC):
    @abstractmethod
    def generate_metric(self, records: List[dict]) -> float | Tensor:
        pass

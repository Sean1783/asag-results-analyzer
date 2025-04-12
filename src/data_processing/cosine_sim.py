from typing import Tuple, List

from sentence_transformers import SentenceTransformer
from torch import Tensor
import torch.nn.functional as F

from src.data_processing.metric_generator import MetricGenerator

class CosineSim(MetricGenerator):
    def __init__(self, embedding_model : str):
        self.embedding_model = embedding_model

    def generate_metric(self, records: List[dict]) -> Tensor:
        ref_ans_embd, std_ans_emb = self.generate_embeddings(records)
        similarities = self.compute_pairwise_similarities(ref_ans_embd, std_ans_emb)
        return similarities

    def generate_embeddings(self, sample_list: list[dict], embedding_model: str=None) -> Tuple[Tensor, Tensor]:
        reference_answer_list = []
        student_answer_list = []
        for sample in sample_list:
            reference_answer_list.append(sample["reference_answer"])
            student_answer_list.append(sample["provided_answer"])
        assert len(reference_answer_list) == len(student_answer_list), "Lists must be the same length"
        consolidated_list = reference_answer_list + student_answer_list
        model = SentenceTransformer(self.embedding_model, trust_remote_code=True)
        embeddings = model.encode(consolidated_list, convert_to_tensor=True, normalize_embeddings=True)
        return embeddings[:len(reference_answer_list)], embeddings[len(reference_answer_list):]

    def compute_pairwise_similarities(self, emb_1: Tensor, emb_2: Tensor) -> Tensor:
        return F.cosine_similarity(emb_1, emb_2, dim=1)
from torch import Tensor

from constants import *

from src.data_access.db_factory import DbFactory
from src.data_processing.metric_context import MetricContext
from src.data_processing.quad_weight_kappa import QuadWeightKappa

DATABASE_NAME = "myergerDB"
NOMIC = "nomic-ai/nomic-embed-text-v1"
MINI_LM = "all-MiniLM-L6-v2"


def add_cosine_similarity_to_record_list(sample_list: list[dict], cosine_similarity_scores: Tensor) -> list[dict]:
    for record, score in zip(sample_list, cosine_similarity_scores):
        record["cosine_similarity"] = score.item()
    return sample_list


def main():
    database_name = DbDetails.MYERGER_DB_NAME.value
    db = DbFactory.get_database_manager(database_name)

    collections = [
        "chatgpt-4o-latest_Beetle",
        "chatgpt-4o-latest_SAF",
        "chatgpt-4o-latest_Mohler",
        "chatgpt-4o-latest_SciEntsBank",
        "claude-3-haiku-20240307_Beetle",
        "claude-3-haiku-20240307_SAF",
        "claude-3-haiku-20240307_Mohler",
        "claude-3-haiku-20240307_SciEntsBank",
        "gpt-4o-mini_Beetle",
        "gpt-4o-mini_SAF",
        "gpt-4o-mini_Mohler",
        "gpt-4o-mini_SciEntsBank",
        "claude-3-7-sonnet-20250219_Beetle",
        "claude-3-7-sonnet-20250219_SAF",
        "claude-3-7-sonnet-20250219_Mohler",
        "claude-3-7-sonnet-20250219_SciEntsBank",
    ]
    for collection in collections:
        samples = list(db.find_documents(collection))
        for sample in samples:
            sample["ai_response"]["score"] = sample["cosine_similarity"]
        data_metric = MetricContext(QuadWeightKappa())
        qwk = data_metric.generate_metric(samples)
        print(f"{collection} - Cosine Similarity QWK: {qwk}")


if __name__ == '__main__':
    main()

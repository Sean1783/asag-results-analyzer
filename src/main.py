from datetime import datetime
from typing import Tuple, List
import csv
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, root_mean_squared_error
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from constants import *
from src.data_access.db_factory import DbFactory
# from src.data_access.myerger_database_manager import MyergerDatabaseManager
from src.data_processing.cosine_sim import CosineSim
from src.data_processing.metric_context import MetricContext
from src.data_processing.metric_generator import MetricGenerator
from src.data_processing.quad_weight_kappa import QuadWeightKappa

DATABASE_NAME = "myergerDB"
NOMIC = "nomic-ai/nomic-embed-text-v1"
MINI_LM = "all-MiniLM-L6-v2"


# def compute_pearson_correlation(scores : Dict[str, List[Any]]) -> pd.DataFrame:
#     df = pd.DataFrame(scores)
#     correlation_matrix = df.corr(method='pearson')
#     print("Pandas Correlation Matrix:")
#     print(correlation_matrix)
#     return correlation_matrix



'''
def compute_squared_error(human_scores: List[float], predictor_scores: List[float]) -> float:
    human_score_list = np.array(human_scores)
    predictor_list = np.array(predictor_scores)
    rmse = root_mean_squared_error(human_score_list, predictor_list)
    return rmse


def compute_squared_error_score(true_score: float, predictor_score: float) -> float:
    return (true_score - predictor_score) ** 2


def compute_average_score_baseline(human_scores: List[float]) -> float:
    return sum(human_scores) / len(human_scores)
'''


'''
def generate_embeddings(sample_list: list[dict], embedding_model: str) -> Tuple[Tensor, Tensor]:
    reference_answer_list = []
    student_answer_list = []
    for sample in sample_list:
        reference_answer_list.append(sample["reference_answer"])
        student_answer_list.append(sample["provided_answer"])
    assert len(reference_answer_list) == len(student_answer_list), "Lists must be the same length"
    consolidated_list = reference_answer_list + student_answer_list
    model = SentenceTransformer(embedding_model, trust_remote_code=True)
    embeddings = model.encode(consolidated_list, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings[:len(reference_answer_list)], embeddings[len(reference_answer_list):]


def compute_pairwise_similarities(emb_1: Tensor, emb_2: Tensor) -> Tensor:
    return F.cosine_similarity(emb_1, emb_2, dim=1)

'''
def add_cosine_similarity_to_record_list(sample_list: list[dict], cosine_similarity_scores: Tensor) -> list[dict]:
    for record, score in zip(sample_list, cosine_similarity_scores):
        record["cosine_similarity"] = score.item()
    return sample_list


'''
def db_batch_update(db: DatabaseManager, db_collection_name: str, updated_sample_list: list[dict]) -> None:
    db.batch_update_cosine_similarity(db_collection_name, updated_sample_list)


def update_collection_records_with_cosine_similarity(
        db: DatabaseManager,
        db_collection_name: str,
        samples: list[dict],
        embedding_model: str) -> None:
    ref_ans_embd, std_ans_emb = generate_embeddings(samples, embedding_model)
    similarities = compute_pairwise_similarities(ref_ans_embd, std_ans_emb)
    updated_sample_list = add_cosine_similarity_to_record_list(samples, similarities)
    db.batch_update_cosine_similarity(db_collection_name, updated_sample_list)
'''

def main():
    # Going to need a collection name for each AI model and each dataset.
    # collection = "Beetle"
    collection = "claude-3-haiku-20240307_Beetle"
    database_name = DbDetails.MYERGER_DB_NAME.value
    db = DbFactory.get_database_manager(database_name)
    # db = MyergerDatabaseManager(database_name)
    samples = list(db.find_documents(collection))
    data_metric = MetricContext(QuadWeightKappa())
    qwk = data_metric.generate_metric(samples)
    print(f"QWK: {qwk}")
    data_metric.set_metric_generator(CosineSim(NOMIC))
    similarities = data_metric.generate_metric(samples)
    # for sample in similarities:
    #     print(sample.item())
    updated_records = add_cosine_similarity_to_record_list(samples, similarities)
    db.batch_update_cosine_similarity(collection, updated_records)

'''
def main():
    # collections = ["Beetle", "SAF", "Mohler", "SciEntsBank"]
    collections = ["SAF"]
    database_name = DbDetails.MYERGER_DB_NAME.value
    db = DatabaseManager(database_name)
    for collection in collections:
        samples = list(db.find_documents(collection))
        # To update db records with cosine similarity - must be run once before generating results
        # update_collection_records_with_cosine_similarity(db, collection, samples, NOMIC)
        human_scores = [sample["normalized_grade"] for sample in samples]
        true_score_average = compute_average_score_baseline(human_scores)
        quadratic_weighted_kappa = run_qwk(samples)

        field_names = [
            "_id",
            "true_score",
            "weight",
            "average_score_baseline",
            "ai_score",
            "cosine_similarity_score",
            "average_score_predictor_error",
            "ai_predictor_error",
            "cosine_similarity_predictor_error",
            "quadratic_weighted_kappa"]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = Path(__file__).parent.parent / "data" / f"{collection}_{timestamp}.csv"
        # filename = f".../data/{collection}_{timestamp}.csv"

        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            for rec in samples:
                _id = rec["_id"]
                weight = rec["weight"]
                true_score = rec["normalized_grade"]
                ai_score = rec["ai_response"]["score"]
                cosine_similarity_score = rec["cosine_similarity"]
                average_score_error = compute_squared_error_score(true_score, true_score_average)
                ai_score_error = compute_squared_error_score(true_score, ai_score)
                cosine_similarity_error = compute_squared_error_score(true_score, cosine_similarity_score)
                writer.writerow({
                    "_id": _id,
                    "true_score": true_score,
                    "weight": weight,
                    "average_score_baseline": true_score_average,
                    "ai_score": ai_score,
                    "cosine_similarity_score": cosine_similarity_score,
                    "average_score_predictor_error": average_score_error,
                    "ai_predictor_error": ai_score_error,
                    "cosine_similarity_predictor_error": cosine_similarity_error,
                    "quadratic_weighted_kappa": quadratic_weighted_kappa
                })


'''


        # Works
        # print("QWK : ", run_qwk(samples))

        # samples = add_cosine_similarity_to_records(samples, NOMIC)

        # cosine_similarity_scores = [sample["cosine_similarity_score"] for sample in samples]
        # data = {"human_scores" : human_scores,
        #         "ai_scores" : ai_scores,
        #         "cosine_similarity_scores" : cosine_similarity_scores}

        # pearson_matrix = compute_pearson_correlation(data)

        # print(f"Output for {collection_name} data results:")
        # ai_rmse = compute_root_mean_squared_error(human_scores, ai_scores)
        # print("AI RMSE:", ai_rmse)
        # cosine_rmse = compute_root_mean_squared_error(human_scores, cosine_similarity_scores)
        # print("Cosine RMSE:", cosine_rmse)
        # average_human_score_list = [compute_average_score_baseline(human_scores)] * len(human_scores)
        # average_score_baseline_rmse = compute_root_mean_squared_error(human_scores, average_human_score_list)
        # print("RMSE of average score predictor:", average_score_baseline_rmse)


if __name__ == '__main__':
    main()

    # QWK results
    # Connected to database
    # Collection : Beetle - AI/human QWK: 0.5888277689527681
    # Connected to database
    # Collection : SAF - AI/human QWK: 0.4075637509986396
    # Connected to database
    # Collection : Mohler - AI/human QWK: 0.7128694718369806
    # Connected to database
    # Collection : SciEntsBank - AI/human QWK: 0.5271823778090929

    # Connected to database
    # Collection : Beetle - AI & human QWK: 0.5888277689527681
    # Connected to database
    # Collection : SAF - AI & human QWK: 0.4075637509986396
    # Connected to database
    # Collection : Mohler - AI & human QWK: 0.7128694718369806
    # Connected to database
    # Collection : SciEntsBank - AI & human QWK: 0.5271823778090929

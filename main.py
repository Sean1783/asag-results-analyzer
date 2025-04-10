from typing import Tuple, List, Dict, Any

from database_manager import DatabaseManager
from sklearn.metrics import cohen_kappa_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util, SimilarityFunction
from nomic import embed
import pandas as pd

from constants import *

import csv

DATABASE_NAME = "myergerDB"
NOMIC = "nomic-ai/nomic-embed-text-v1"
MINI_LM = "all-MiniLM-L6-v2"

# def compute_pearson_correlation(scores : Dict[str, List[Any]]) -> pd.DataFrame:
#     df = pd.DataFrame(scores)
#     correlation_matrix = df.corr(method='pearson')
#     print("Pandas Correlation Matrix:")
#     print(correlation_matrix)
#     return correlation_matrix

# def normalize_cosine_similarity_scores(cosine_similarity_scores : List[float]):
#     grade_min, grade_max = 0.0, 1.0
#     cosine_similarities = np.array(cosine_similarity_scores).reshape(-1, 1)
#     scaler = MinMaxScaler(feature_range=(grade_min, grade_max))
#     normalized_grades = scaler.fit_transform(cosine_similarities)
#     return normalized_grades.flatten()

# def convert_to_embedding(sentence_one, sentence_two, embedding_model):
#     model = SentenceTransformer(embedding_model, trust_remote_code=True)
#     embedding = model.encode([sentence_one, sentence_two], convert_to_tensor=True, normalize_embeddings=True)
#     return embedding


# def compute_cosine_similarity(tensor_obj) -> float:
#     return util.pytorch_cos_sim(tensor_obj[0], tensor_obj[1]).item()

# def get_all_records(database : str, collection : str) -> List[dict]:
#     db = DatabaseManager(database)
#     db_collection = db.get_db()[collection]
#     return list(db_collection.find())
#     # return list(db_collection.find().limit(10))

# def add_cosine_similarity_to_records(samples : List[dict], embedding_model : str) -> List[dict]:
#     for record in samples:
#         reference_answer = record["reference_answer"]
#         provided_answer = record["provided_answer"]
#         embeddings = convert_to_embedding(reference_answer, provided_answer, embedding_model)
#         cosine_similarity = compute_cosine_similarity(embeddings)
#         record["cosine_similarity_score"] = cosine_similarity
#     return samples


def compute_root_mean_squared_error(human_scores : List[float], predictor_scores : List[float]) -> float:
    human_score_list = np.array(human_scores)
    predictor_list = np.array(predictor_scores)
    rmse = root_mean_squared_error(human_score_list, predictor_list)
    return rmse

def compute_average_score_baseline(human_scores : List[float]) -> float:
    return sum(human_scores) / len(human_scores)

def generate_binned_values_map(feature_name : str, scores : List[float]) -> Dict[str, int]:
    field_name = "discrete_" + feature_name
    binned_scores = {field_name: [round(score * 10) for score in scores]}
    return binned_scores

def compute_quadratic_weighted_kappa(human_scores_list : List[float], predictor_score_list: List[float]) -> float:
    discrete_human_scores = [round(score * 10) for score in human_scores_list]
    discrete_predictor_scores = [round(score * 10) for score in predictor_score_list]
    predictor_qwk = cohen_kappa_score(discrete_human_scores, discrete_predictor_scores, weights="quadratic" )
    return predictor_qwk

def run_qwk(sample_list : list[dict]) -> float:
    human_scores = [sample["normalized_grade"] for sample in sample_list]
    ai_scores = [sample["ai_response"]["score"] for sample in sample_list]
    qwk = compute_quadratic_weighted_kappa(human_scores, ai_scores)
    return qwk

# def generate_embeddings(
#         reference_answer_list : List[str],
#         student_answer_list : List[str],
#         embedding_model : str) -> Tuple[Tensor, Tensor]:
#     pass
#     # assert len(reference_answer_list) == len(student_answer_list), "Lists must be the same length"
#     # consolidated_list = reference_answer_list + student_answer_list
#     # model = SentenceTransformer(embedding_model, trust_remote_code=True)
#     # embeddings = model.encode(consolidated_list, convert_to_tensor=True, normalize_embeddings=True)
#     # return embeddings[:len(reference_answer_list)], embeddings[len(reference_answer_list):]

def generate_embeddings(sample_list : list[dict], embedding_model : str) -> Tuple[Tensor, Tensor]:
    # reference_answer_list = [sample["reference_answer"] for sample in sample_list]
    # student_answer_list = [sample["provided_answer"] for sample in sample_list]
    reference_answer_list = []
    student_answer_list = []
    for sample in sample_list:
        reference_answer_list.append(sample["reference_answer"])
        student_answer_list.append(sample["provided_answer"])
    assert len(reference_answer_list) == len(student_answer_list), "Lists must be the same length"
    consolidated_list = reference_answer_list + student_answer_list
    model = SentenceTransformer(embedding_model, trust_remote_code=True)
    embeddings = model.encode(consolidated_list, convert_to_tensor=True, normalize_embeddings=True)
    # return generate_embeddings(reference_answer, student_answer, embedding_model)
    return embeddings[:len(reference_answer_list)], embeddings[len(reference_answer_list):]

def compute_pairwise_similarities(emb_1: Tensor, emb_2: Tensor) -> Tensor:
    return F.cosine_similarity(emb_1, emb_2, dim=1)

def add_cosine_similarity_to_record_list(
        sample_list : list[dict],
        cosine_similarity_scores : Tensor) -> list[dict]:
    for record, score in zip(sample_list, cosine_similarity_scores):
        record["cosine_similarity"] = score.item()
    return sample_list

def db_batch_update(db : DatabaseManager, db_collection_name : str, updated_sample_list : list[dict]) -> None:
    db.batch_update_cosine_similarity(db_collection_name, updated_sample_list)

def update_collection_records_with_cosine_similarity(
        db : DatabaseManager,
        db_collection_name : str,
        samples : list[dict],
        embedding_model : str) -> None:
    ref_ans_embd, std_ans_emb = generate_embeddings(samples, embedding_model)
    similarities = compute_pairwise_similarities(ref_ans_embd, std_ans_emb)
    updated_sample_list = add_cosine_similarity_to_record_list(samples, similarities)
    db.batch_update_cosine_similarity(db_collection_name, updated_sample_list)


def main():
    collections = ["Beetle", "SAF", "Mohler", "SciEntsBank"]
    database_name = DbDetails.MYERGER_DB_NAME.value
    db = DatabaseManager(database_name)
    for collection in collections:
        samples = list(db.find_documents(collection))
        # update_collection_records_with_cosine_similarity(db, collection, samples, NOMIC)





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

    # Using all-MiniLM-L6-v2 embedding:
    # Using gpt-4o-mini results:

    # Output for Beetle: data results
    # AI RMSE: 0.27306897785480255
    # Cosine RMSE: 0.3315366854401315
    # RMSE of average score predictor: 0.3295264091404324

    # Output for SAF: data results:
    # AI RMSE: 0.26536782648492513
    # Cosine RMSE: 0.30605616690427473
    # RMSE of average score predictor: 0.30531565587473924

    # Output for Mohler: data results:
    # AI RMSE: 0.17185265006200323
    # Cosine RMSE: 0.27780096999831605
    # RMSE of average score predictor: 0.2299939612733826

    # Output for SciEntsBank: data results:
    # AI RMSE: 0.3262695999504886
    # Cosine RMSE: 0.3741241689943224
    # RMSE of average score predictor: 0.38351686427742027


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

from typing import Tuple, List, Dict, Any

from database_manager import DatabaseManager
from sklearn.metrics import cohen_kappa_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from sentence_transformers import SentenceTransformer, util, SimilarityFunction
from nomic import embed
import pandas as pd

import csv

DATABASE_NAME = "myergerDB"
NOMIC = "nomic-ai/nomic-embed-text-v1"
MINI_LM = "all-MiniLM-L6-v2"

def compute_pearson_correlation(scores : Dict[str, List[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(scores)
    correlation_matrix = df.corr(method='pearson')
    print("Pandas Correlation Matrix:")
    print(correlation_matrix)
    return correlation_matrix

def normalize_cosine_similarity_scores(cosine_similarity_scores : List[float]):
    grade_min, grade_max = 0.0, 1.0
    cosine_similarities = np.array(cosine_similarity_scores).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(grade_min, grade_max))
    normalized_grades = scaler.fit_transform(cosine_similarities)
    return normalized_grades.flatten()

def convert_to_embedding(sentence_one, sentence_two, embedding_model):
    model = SentenceTransformer(embedding_model, trust_remote_code=True)
    embedding = model.encode([sentence_one, sentence_two], convert_to_tensor=True)
    return embedding

def compute_cosine_similarity(tensor_obj) -> float:
    return util.pytorch_cos_sim(tensor_obj[0], tensor_obj[1]).item()

'''
Correct this to return the entire set from the database ...'limit()' <-- remove this
'''
def get_all_records(database : str, collection : str) -> List[dict]:
    db = DatabaseManager(database)
    db_collection = db.get_db()[collection]
    return list(db_collection.find().limit(50))

def add_cosine_similarity_to_records(samples : List[dict], embedding_model : str) -> List[dict]:
    for record in samples:
        reference_answer = record["reference_answer"]
        provided_answer = record["provided_answer"]
        embeddings = convert_to_embedding(reference_answer, provided_answer, embedding_model)
        cosine_similarity = compute_cosine_similarity(embeddings)
        record["cosine_similarity_score"] = cosine_similarity
    return samples

def generate_binned_values_map(feature_name : str, scores : List[float]) -> Dict[str, int]:
    field_name = "discrete_" + feature_name
    binned_scores = {field_name: [round(score * 10) for score in scores]}
    return binned_scores

def compute_root_mean_squared_error(human_scores : List[float], comparison_scores : List[float]) -> float:
    human = np.array(human_scores)
    comparison = np.array(comparison_scores)
    rmse = root_mean_squared_error(human, comparison)
    return rmse

def main():
    samples = get_all_records(DATABASE_NAME, "ds1")
    samples = add_cosine_similarity_to_records(samples, MINI_LM)
    human_scores = [sample["normalized_grade"] for sample in samples]
    ai_scores = [sample["ai_response"]["score"] for sample in samples]
    cosine_similarity_scores = [sample["cosine_similarity_score"] for sample in samples]
    data = {"human_scores" : human_scores,
            "ai_scores" : ai_scores,
            "cosine_similarity_scores" : cosine_similarity_scores}
    # pearson_matrix = compute_pearson_correlation(data)

    # QWK section
    # discrete_human = [round(score * 10) for score in human_scores]
    # discrete_ai = [round(score * 10) for score in ai_scores]
    # discrete_cosine = [round(score * 10) for score in cosine_similarity_scores]
    # ai_qwk = cohen_kappa_score(discrete_human, discrete_ai, weights="quadratic" )
    # cosine_qwk = cohen_kappa_score(discrete_human, discrete_cosine, weights="quadratic")
    # print(f"Quadratic Weighted Kappa: {ai_qwk:.3f}")
    # print(f"Quadratic Weighted Kappa: {cosine_qwk:.3f}")

    ai_rmse = compute_root_mean_squared_error(human_scores, ai_scores)
    print("AI RMSE:", ai_rmse)
    cosine_rmse = compute_root_mean_squared_error(human_scores, cosine_similarity_scores)
    print("Cosine RMSE:", cosine_rmse)


if __name__ == '__main__':
    main()
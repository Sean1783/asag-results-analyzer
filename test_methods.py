from typing import Tuple, List, Dict, Any

from database_manager import DatabaseManager
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from sentence_transformers import SentenceTransformer, util, SimilarityFunction
from nomic import embed
import pandas as pd

import csv

DATABASE_NAME = "myergerDB"
NOMIC = "nomic-ai/nomic-embed-text-v1"  # --- for nomic embedding
MINI_LM = "all-MiniLM-L6-v2"  # --- for simple fast embedding model


def connect_to_database(database_name: str) -> None | DatabaseManager:
    db = DatabaseManager(database_name)
    return db

    # DATABASE_NAME = "myergerDB"
    # DATABASE_COLLECTION = "results1"


def get_all_documents(collection):
    db = DatabaseManager("myergerDB")
    records = db.find_documents(collection)
    aggregate_human_score = 0
    aggregate_ai_score = 0
    total_records = 0
    for record in records:
        print(record)
        aggregate_human_score += record["normalized_grade"]
        aggregate_ai_score += record["ai_response"]["score"]
        total_records += 1
    print("Mean human grade : " + str(aggregate_human_score / total_records))
    print("Mean AI grade : " + str(aggregate_ai_score / total_records))


def cosine_similarity_test() -> dict[str, list[Any]]:
    db = DatabaseManager("myergerDB")
    records = db.find_documents("results1")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    human_scores = []
    ai_scores = []
    cosine_similarity_scores = []

    for record in records:
        reference_answer = record["reference_answer"]
        student_answer = record["provided_answer"]
        embeddings = model.encode([reference_answer, student_answer], convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        human_scores.append(record["normalized_grade"])
        ai_scores.append(record["ai_response"]["score"])
        cosine_similarity_scores.append(similarity_score)
    data = {"human_scores": human_scores, "ai_scores": ai_scores, "cosine_similarity_scores": cosine_similarity_scores}
    return data
    # record["similarity_score"] = similarity_score
    # data = {}
    # data["reference_answer"] = reference_answer
    # data["student_answer"] = student_answer
    # data["ai_model"] = record["ai_model"]
    # data["normalized_grade"] = record["normalized_grade"]
    # data["ai_score"] = record["ai_response"]["score"]
    # data["similarity_score"] = similarity_score
    # samples.append(data)

    # field_names = samples[0].keys()
    # with open('results1.csv', 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=field_names)
    #     writer.writeheader()
    #     writer.writerows(samples)


def nomic_test():
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

    sentences = [
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day",
        "Yesterday was a sunny day",
        "The dog is drinking beer"
    ]
    embeddings = model.encode(sentences)

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    # print(embeddings)


def compare_embedding_similarities():
    db = DatabaseManager("myergerDB")
    records = db.find_documents("results1", {'data_source': 'SciEntsBank'})
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    nomic_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    cs_scores = {}
    d = {}
    for record in records:
        reference_answer = record["reference_answer"]
        student_answer = record["provided_answer"]
        sbert_embedding = sbert_model.encode([reference_answer, student_answer], convert_to_tensor=True)
        # momic_embedding = nomic_model.encode([reference_answer, student_answer], convert_to_tensor=True)
        sbert_similarity_score = util.pytorch_cos_sim(sbert_embedding[0], sbert_embedding[1]).item()
        # nomic_similarity_score = util.pytorch_cos_sim(momic_embedding[0], momic_embedding[1]).item()
        result = {}
        result["source"] = record["data_source"]
        result["sbert_similarity_score"] = sbert_similarity_score
        # result["nomic_similarity_score"] = nomic_similarity_score
        result["ai_score"] = record["ai_response"]["score"]
        result["normalized_grade"] = record["normalized_grade"]
        d[record["Index"]] = result
        cs_scores[record["Index"]] = result

        # grade_min, grade_max = 0.0, 1.0
        # cosine_similarities = np.array([sbert_similarity_score, nomic_similarity_score]).reshape(-1, 1)
        # scaler = MinMaxScaler(feature_range=(grade_min, grade_max))
        # normalized_grades = scaler.fit_transform(cosine_similarities)
        # print(normalized_grades.flatten())
        # print(record["normalized_grade"], record["ai_response"]["score"])


def something():
    db = DatabaseManager("myergerDB")
    collection = db.get_db()["ds1"]
    dataset_names = collection.distinct("data_source")
    question_bank = {}

    for dataset_name in dataset_names:
        question_set = list(collection.find({"data_source": dataset_name}))
        question_bank[dataset_name] = question_set

    cosine_similarity_scores = []
    for dataset in question_bank:
        for record in question_bank[dataset]:
            reference_answer = record["reference_answer"]
            provided_answer = record["provided_answer"]
            embeddings = convert_to_embedding(reference_answer, provided_answer, MINI_LM)
            cosine_similarity = compute_cosine_similarity(embeddings)
            record["cosine_similarity_score"] = cosine_similarity
            cosine_similarity_scores.append(cosine_similarity)
        rescaled_cosine_similarity = normalize_cosine_similarity_scores(cosine_similarity_scores)
        i = 0
        for record in question_bank[dataset]:
            record["rescaled_cosine_similarity_score"] = rescaled_cosine_similarity[i]
            i += 1

    for dataset in question_bank:
        for record in question_bank[dataset]:
            print("AI score : " + str(record["ai_response"]["score"]),
                  "Human score : " + str(record["normalized_grade"]),
                  "Cosine similarity : " + str(record["cosine_similarity_score"]),
                  "Rescaled cosine similarity : " + str(record["rescaled_cosine_similarity_score"]),
                  "Reference answer : " + record["reference_answer"],
                  "Provided answer : " + record["provided_answer"])


def prep_for_pearson(samples: List[dict]) -> dict[str, list[Any]]:
    human_scores = [sample["normalized_grade"] for sample in samples]
    ai_scores = [sample["ai_response"]["score"] for sample in samples]
    cosine_similarity_scores = [sample["cosine_similarity_score"] for sample in samples]
    normalized_cosine_sim_scores = [sample["normalized_cosine_similarity_score"] for sample in samples]
    data = {"human_scores": human_scores,
            "ai_scores": ai_scores,
            "cosine_similarity_scores": cosine_similarity_scores,
            "normalized_cosine_sim_scores": normalized_cosine_sim_scores}
    return data


def compute_pearson_correlation(scores: Dict[str, List[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(scores)
    correlation_matrix = df.corr(method='pearson')
    print("Pandas Correlation Matrix:")
    print(correlation_matrix)
    return correlation_matrix


def normalize_cosine_similarity_scores(cosine_similarity_scores: List[float]):
    grade_min, grade_max = 0.0, 1.0
    cosine_similarities = np.array(cosine_similarity_scores).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(grade_min, grade_max))
    normalized_grades = scaler.fit_transform(cosine_similarities)
    return normalized_grades.flatten()


def convert_to_embedding(sentence_one, sentence_two, embedding_model):
    model = SentenceTransformer(embedding_model)
    embedding = model.encode([sentence_one, sentence_two], convert_to_tensor=True)
    return embedding


def compute_cosine_similarity(tensor_obj) -> float:
    return util.pytorch_cos_sim(tensor_obj[0], tensor_obj[1]).item()


def get_all_records(database: str, collection: str) -> List[dict]:
    db = DatabaseManager(database)
    db_collection = db.get_db()[collection]
    return list(db_collection.find())


def add_cosine_similarity_to_records(samples: List[dict], embedding_model: str) -> List[dict]:
    for record in samples:
        reference_answer = record["reference_answer"]
        provided_answer = record["provided_answer"]
        embeddings = convert_to_embedding(reference_answer, provided_answer, embedding_model)
        cosine_similarity = compute_cosine_similarity(embeddings)
        record["cosine_similarity_score"] = cosine_similarity
    return samples


# Could use Pearson correlation to illustrate the trends that the AI captures in grading.

def main_function():
    samples = get_all_records(DATABASE_NAME, "ds1")
    samples = add_cosine_similarity_to_records(samples, NOMIC)
    cos_sim_scores = [sample["cosine_similarity_score"] for sample in samples]
    normalized_cosine_similarity_scores = normalize_cosine_similarity_scores(cos_sim_scores)
    for i in range(len(samples)):
        samples[i]["normalized_cosine_similarity_score"] = normalized_cosine_similarity_scores[i]
    data = prep_for_pearson(samples)
    pearson_matrix = compute_pearson_correlation(data)




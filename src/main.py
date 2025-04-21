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
from src. data_output.myerger_csv_exporter import MyergerCsvExporter
from src.data_processing.cosine_sim import CosineSim
from src.data_processing.metric_context import MetricContext
from src.data_processing.metric_generator import MetricGenerator
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

    # For quickly recomputing all QWK values.
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
        data_metric = MetricContext(QuadWeightKappa())
        qwk = data_metric.generate_metric(samples)
        print(f"{collection} - QWK: {qwk}")


    # Done
    # collection = "chatgpt-4o-latest_Beetle"
    # collection = "chatgpt-4o-latest_SAF"
    # collection = "chatgpt-4o-latest_Mohler"
    # collection = "chatgpt-4o-latest_SciEntsBank"
    # collection = "claude-3-haiku-20240307_Beetle"
    # collection = "claude-3-haiku-20240307_SAF"
    # collection = "claude-3-haiku-20240307_Mohler"
    # collection = "claude-3-haiku-20240307_SciEntsBank"
    # collection = "gpt-4o-mini_Beetle"
    # collection = "gpt-4o-mini_SAF"
    # collection = "gpt-4o-mini_Mohler"
    # collection = "gpt-4o-mini_SciEntsBank"
    # collection=  "claude-3-7-sonnet-20250219_Beetle"
    # collection = "claude-3-7-sonnet-20250219_SAF"
    # collection = "claude-3-7-sonnet-20250219_Mohler"
    # collection = "claude-3-7-sonnet-20250219_SciEntsBank"

    # samples = list(db.find_documents(collection))
    # data_metric = MetricContext(QuadWeightKappa())
    # qwk = data_metric.generate_metric(samples)
    # print(f"{collection} - QWK: {qwk}")
    # data_metric.set_metric_generator(CosineSim(NOMIC))
    # similarities = data_metric.generate_metric(samples)
    # updated_records = add_cosine_similarity_to_record_list(samples, similarities)
    # db.batch_update_cosine_similarity(collection, updated_records)

if __name__ == '__main__':
    main()
import os
from typing import Any, List

from pymongo import MongoClient
from dotenv import load_dotenv
from pymongo.synchronous.cursor import Cursor, Mapping

class DatabaseManager:
    def __init__(self, database_name: str) -> None:
        self.db = None
        self.connect_to_database(database_name)

    def get_db(self):
        if self.db is None:
            self.connect_to_database(os.getenv("MONGO_DATABASE"))
            return self.db
        else:
            return self.db

    def connect_to_database(self, db_name : str) -> None:
        load_dotenv()
        uri = os.getenv("MONGO_URI")
        try:
            client = MongoClient(uri, tls=True, tlsCAFile='/etc/ssl/cert.pem')
            self.db = client[db_name]
            print("Connected to database")
        except Exception as e:
            print(f"Database did not connect successfully : {e}")

    def insert_document(self, collection, document) -> int | None:
        try:
            collection = self.db[collection]
            result = collection.insert_one(document)
            return result.inserted_id
        except Exception as e:
            print(f"Database did not insert document successfully : {e}")

    def insert_documents(self, collection_name : str, documents : List) -> List | None:
        try:
            collection = self.db[collection_name]
            result = collection.insert_many(documents)
            return result.inserted_ids
        except Exception as e:
            print(f"Database did not insert documents successfully : {e}")

    def find_documents(self, collection, query=None) -> Cursor[Mapping[str, Any] | Any] | None:
        if query is None:
            query = {}
        try:
            collection = self.db[collection]
            results = collection.find(query)
            return results
        except Exception as e:
            print(f"Database did not find documents successfully : {e}")
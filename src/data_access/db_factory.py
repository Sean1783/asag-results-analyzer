from src.data_access.myerger_db_manager import MyergerDbManager


class DbFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_database_manager(database_name: str) -> MyergerDbManager:
        if database_name == "myergerDB":
            return MyergerDbManager(database_name)
        else:
            raise Exception(f"Database name {database_name} does not exist")

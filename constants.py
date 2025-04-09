from enum import Enum

# class AIModels(Enum):
#     GPT_4O_MINI = "gpt-4o-mini"
#     CHATGPT_4O_LATEST = "chatgpt-4o-latest"
#     CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
#     CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
#     GEMINI_2_FLASH = "gemini-2.0-flash"

class FeatureNameValues(Enum):
    QUESTION = "question"
    ANSWER = "answer"
    REFERENCE_ANSWER = "reference_answer"
    AI_SCORE = "score"

class SampleStrategyNames(Enum):
    RANDOM = "random"
    BY_FEATURE = "by_feature"

class DbDetails(Enum):
    MYERGER_DB_NAME = "myergerDB"
    DB_COLLECTION_BEETLE = "Beetle"
    DB_COLLECTION_SAF = "SAF"
    DB_COLLECTION_MOHLER = "Mohler"
    DB_COLLECTION_SCIENTSBANK = "SciEntsBank"


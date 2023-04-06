import os
import inspect
import fnmatch
from collections import OrderedDict

base_path, current_dir = os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))

class Config(object):
    QDEBUG = True
    
    FILES = dict(
        DATA_LOCAL_DIR  = os.path.join(base_path, "data"),
        PREPROCESS_DIR  = os.path.join(base_path, "data", "preprocess"),
        PDF_DIR         = os.path.join(base_path, "data", "pdf"),
        DATABASE_DIR    = os.path.join(base_path, "data", "database", "Text.db"),
        MODEL_DATA_DIR  = os.path.join(base_path, "data", "models"),
        
        RAW_NEWS_SUMMARY_FILE       = "news_summary",
        PROCESS_NEWS_SUMMARY_FILE   = "process_news_summary",
        
    )
    
    ANALYSIS_CONFIG = dict(
        SPLIT_RATIO                 = 0.1,
        TEXT_MAX_TOKEN_LEN          = 512,
        SUMMARY_MAX_TOKEN_LEN       = 128,
        BATCH_SIZE                  = 8,
        
        MODEL_MAX_LENGTH            = 512,
        TOKENIZER_PADDING           = "max_length",
        TOKENIZER_TRUNCATION        = True,
        TOKENIZER_ATTENTION_MASK    = True,
        TOKENIZER_ADD_SPECIAL_TOKEN = True,
        TOKENIZER_RETURN_TENSORS    = "pt",
        ZERO_INPUT_IDS              = -100,
        
        
    )
    
    
    MODELLING_CONFIG = dict(
        N_EPOCHS        = 3,
        BATCH_SIZE      = 8,
        MODEL_NAME      = "t5-base",
        EXT_RATIO       = 1.0,
        MIN_LENGTH      = 80,
        MAX_LENGTH      = 150,
        LENGTH_PENALTY  = 15,
        NUM_BEAMS       = 2,
    )
    
    
    VARS = OrderedDict(
        REVIEWS = [
            dict(var="rating",          dtypes=float,   predictive=False),
            dict(var="page_number",     dtypes=int,     predictive=True),
            dict(var="book_author",     dtypes=str,     predictive=True),
            dict(var="reviews",         dtypes=str,     predictive=True),
            dict(var="title",           dtypes=str,     predictive=True),
            dict(var="reviews_length",  dtypes=int,     predictive=True),
        ],
        
    )
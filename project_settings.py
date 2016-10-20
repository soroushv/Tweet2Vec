__author__ = 'pralav'
import os
DATASETS='datasets'
MODELS='models'
INTER='intermediate'
OUTPUTS='outputs'
QUERY_RULES='query_rules'
TESTS='tests'
TWEET2VEC='tweet2vec'
BASE_DIR=os.path.dirname(os.path.realpath(__file__))
BASE_DATA_PATH=os.path.join(BASE_DIR,"all_data")
GLOBAL_PARAMS_PATH=os.path.join(BASE_DIR,"global_model_params")
LOG_PATH=BASE_DIR+"/logs"
DATA_TYPES=[MODELS,DATASETS,INTER,OUTPUTS,TESTS,QUERY_RULES]
MODULES=[TWEET2VEC]
LOGGING_CONFIG=os.path.join(BASE_DIR,'configs/logging.json')

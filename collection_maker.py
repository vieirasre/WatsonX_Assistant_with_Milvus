import os
import logging
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Parâmetros de conexão
MILVUS_HOST = os.environ.get("REMOTE_SERVER", '127.0.0.1')
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
INDEX_NAME = "ML_Collection2"
EMBEDDING_DIM = 768

# Configurar logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info("Logger initialized")

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.info(f"Dropped existing collection: {collection_name}")

    fields = [
        FieldSchema(name='pk', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65_535),
        FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    
    schema = CollectionSchema(fields=fields, description="General collection for id, text and vector")
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    logger.info(f"Created collection: {collection_name}")
    return collection

if __name__ == "__main__":
    logger.info("Starting the Milvus connection process")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    
    if not utility.has_collection(INDEX_NAME):
        logger.info(f"Creating collection {INDEX_NAME}")
        create_milvus_collection(INDEX_NAME, dim=EMBEDDING_DIM)
    else:
        logger.info(f"Collection {INDEX_NAME} already exists")

import os
import logging
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Parâmetros de conexão
MILVUS_HOST = os.environ.get("REMOTE_SERVER", '127.0.0.1')
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
INDEX_NAME = "ML_Collection"
EMBEDDING_DIM = 768

# Configurar logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levellevelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Logger initialized")

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.info(f"Dropped existing collection: {collection_name}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    
    schema = CollectionSchema(fields=fields, description="General collection for embeddings and metadata")
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info(f"Created collection: {collection_name}")
    return collection

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    
    if not utility.has_collection(INDEX_NAME):
        logging.info(f"Creating collection {INDEX_NAME}")
        create_milvus_collection(INDEX_NAME, dim=EMBEDDING_DIM)
    else:
        logging.info(f"Collection {INDEX_NAME} already exists")


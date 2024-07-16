import os
import logging
import PyPDF2
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.foundation_models import Model
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Informações dos arquivos para inserir na coleção (mdswift)
SOURCE_FILE_NAMES = ["Apostila_ML.pdf", "Artigo_ML.pdf"]
SOURCE_URLS = ["https://github.com/vieirasre/WatsonX_Assistant_with_Milvus/blob/main/Apostila_ML.pdf", 
               "https://github.com/vieirasre/WatsonX_Assistant_with_Milvus/blob/main/Artigo_ML.pdf"]
SOURCE_TITLES = ["Apostila Machine Learning UFES", "Artigo Machine Learning UE"]
SOURCES_TOPIC = "Conteúdos Machine Learning"
INDEX_NAME = "ML_Collection_to_LC_teste2"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 20

# Parâmetros de conexão
MILVUS_HOST = os.environ.get("REMOTE_SERVER", '127.0.0.1')
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_CONNECTION = {"host": MILVUS_HOST, "port": MILVUS_PORT}

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
        FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=1536)
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

def connect_watsonx():
    API_KEY = os.environ.get("WATSONX_APIKEY")
    PROJECT_ID = os.environ.get("PROJECT_ID")
    URL = "https://us-south.ml.cloud.ibm.com"

    wml_credentials = {
        "url": URL,
        "apikey": API_KEY,
    }

    client = APIClient(wml_credentials)
    client.set.default_project(PROJECT_ID)
    model = Model(client, "granite-1")
    
    logger.info("Successfully connected to WatsonX")
    
    return model

def embed_text(text):
    model = connect_watsonx()
    embedding = model.generate_embeddings(text)
    return embedding

def load_docs_pdf(filenames, urls, titles):
    texts = []
    metadata = []
    for i, filename in enumerate(filenames):
        url = urls[i] if i < len(urls) else ""
        title = titles[i] if i < len(titles) else ""
        try:
            with open(filename, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    texts.append(text)
                    metadata.append({'url': url, 'title': title})
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
    return texts, metadata

def index_documents(connection_info, filenames, urls, titles):
    texts, metadata = load_docs_pdf(filenames, urls, titles)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_texts = text_splitter.create_documents(texts, metadata)
    logger.info(f"Documents chunked. Sending to Milvus.")
    try:
        collection = create_milvus_collection()
        index = Milvus.from_documents(documents=split_texts, embedding=embed_text, connection_args=connection_info, collection_name=INDEX_NAME)
        return index
    except Exception as e:
        logger.error(f"Failed to index documents: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting the Milvus connection process")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    index = index_documents(MILVUS_CONNECTION, SOURCE_FILE_NAMES, SOURCE_URLS, SOURCE_TITLES)

    query = "What is Data Mining?"
    try:
        results = index.similarity_search(query)
        print(results)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")

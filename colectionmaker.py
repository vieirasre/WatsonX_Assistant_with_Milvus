import time
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import os
import PyPDF2
import logging
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.foundation_models import Model
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter

SOURCE_FILE_NAMES = ["Apostila ML.pdf", "Artigo ML.pdf"]
SOURCE_URLS = ["https://petmecanica.ufes.br/sites/petengenhariamecanica.ufes.br/files/field/anexo/apostila_do_minicurso_de_machine_learning.pdf", 
               "https://dspace.uevora.pt/rdpc/bitstream/10174/30174/1/apra_techReport_paper01_2018-11_v02.pdf"]
SOURCE_TITLES = ["Apostila Machine Learning UFES", "Artigo Machine Learning UE"]
SOURCES_TOPIC = "Conteúdos Machine Learning"
INDEX_NAME = "ML"

MILVUS_CONNECTION = {"host": os.environ.get("REMOTE_SERVER", '127.0.0.1'), 
                     "port": '19530'}

CHUNK_SIZE = 250
CHUNK_OVERLAP = 20

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Logger initialized")

def connect_watsonx():
    API_KEY = os.environ.get("WATSONX_API_KEY")
    PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
    URL = "https://us-south.ml.cloud.ibm.com"

    wml_credentials = {
        "url": URL,
        "apikey": API_KEY,
    }

    client = APIClient(wml_credentials)
    client.set.default_project(PROJECT_ID)
    model = Model(client, "granite-1")
    
    return model

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

# Defina o nome da coleção e a dimensão dos embeddings
COLLECTION_NAME = "Collection_ML"
EMBEDDING_DIM = 768  

# Cria a coleção
create_milvus_collection(COLLECTION_NAME, EMBEDDING_DIM)

def connect(connection_info):
    index = Milvus(
        embedding_function=embed_text,  # Esta função será definida mais adiante
        connection_args=connection_info,
        collection_name=INDEX_NAME,
        index_params="text"
    )
    return index


def load_docs_pdf(filenames, urls, titles):
    texts = []
    metadata = []
    i = 0
    for filename in filenames:
        if len(urls) > i:
            url = urls[i]
        else:
            url = ""
        if len(titles) > i:
            title = titles[i]
        else:
            title = ""
        with open(filename, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text = page.extract_text()
                texts.append(text)
                metadata.append({'url': url, 'title': title})
    return texts, metadata

def embed_text(text):
    model = connect_watsonx()
    embedding = model.generate_embeddings(text)
    return embedding

def index(connection_info, filenames, urls, titles):
    texts, metadata = load_docs_pdf(filenames, urls, titles)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_texts = text_splitter.create_documents(texts, metadata)
    logging.info(f"Documents chunked. Sending to Milvus.")
    index = Milvus.from_documents(documents=split_texts, embedding=embed_text, connection_args=connection_info, collection_name=INDEX_NAME)
    return index

INDEXED = True

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    if INDEXED:
        logging.info(f"Connecting to {MILVUS_CONNECTION}")
        index = connect(MILVUS_CONNECTION)
    else:
        logging.info(f"Indexing at {MILVUS_CONNECTION}")
        index = index(MILVUS_CONNECTION, SOURCE_FILE_NAMES, SOURCE_URLS, SOURCE_TITLES)
    
    print(index)
    query = "What is Data Mining?"
    results = index.similarity_search(query)
    print(results)


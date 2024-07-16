import os
import logging
import PyPDF2
from pymilvus import connections
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.foundation_models import Model
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Informações dos arquivos para inserir na coleção
SOURCE_FILE_NAMES = ["Apostila ML.pdf", "Artigo ML.pdf"]
SOURCE_URLS = ["https://petmecanica.ufes.br/sites/petengenhariamecanica.ufes.br/files/field/anexo/apostila_do_minicurso_de_machine_learning.pdf", 
               "https://dspace.uevora.pt/rdpc/bitstream/10174/30174/1/apra_techReport_paper01_2018-11_v02.pdf"]
SOURCE_TITLES = ["Apostila Machine Learning UFES", "Artigo Machine Learning UE"]
INDEX_NAME = "ML_Collection"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 20

# Parâmetros de conexão
MILVUS_HOST = os.environ.get("REMOTE_SERVER", '127.0.0.1')
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_CONNECTION = {"host": MILVUS_HOST, "port": MILVUS_PORT}

# Configurar logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levellevelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Logger initialized")

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
    
    print("Successfully connected to WatsonX")
    
    return model

def embed_text(text):
    model = connect_watsonx()
    embedding = model.generate_embeddings(text)
    return embedding

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

def index(connection_info, filenames, urls, titles):
    texts, metadata = load_docs_pdf(filenames, urls, titles)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_texts = text_splitter.create_documents(texts, metadata)
    logging.info(f"Documents chunked. Sending to Milvus.")
    index = Milvus.from_documents(documents=split_texts, embedding=embed_text, connection_args=connection_info, collection_name=INDEX_NAME)
    return index

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    
    logging.info(f"Indexing at {MILVUS_CONNECTION}")
    index = index(MILVUS_CONNECTION, SOURCE_FILE_NAMES, SOURCE_URLS, SOURCE_TITLES)
    
    query = "What is Data Mining?"
    results = index.similarity_search(query)
    print(results)

import os
import logging
import PyPDF2
from pymilvus import connections
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
INDEX_NAME = "ML_Collection_to_LC"

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

def connect(connection_info):
    index = Milvus(
        embed_text,
        connection_args=connection_info,
        collection_name=INDEX_NAME,
        index_params="text"
    )
    return index

def connect_watsonx():
    API_KEY = os.environ.get("WATSONX_APIKEY")
    PROJECT_ID = os.environ.get("PROJECT_ID")
    
    wml_credentials = {
        "url": "https://jp-tok.ml.cloud.ibm.com",
        "apikey": API_KEY,
    }

    client = APIClient(wml_credentials)
    client.set.default_project(PROJECT_ID)
    model = Model(client, "granite-1")
    
    logger.info("Successfully connected to WatsonX")
    
    return model

#def embed_documents(text):
#    model = connect_watsonx()
#    embedding = model.generate_embeddings(text)
#    return embedding

def embed_documents(texts):
    embeddings = []
    model = connect_watsonx()
    logger.info(f"Connected to WatsonX model: {model}")
    for text in texts:
        embedding = model.generate_embeddings(text)
        #logger.info(f"Generated embedding for text: {text[:50]}...")  # Exemplo de logging de partes do texto
        embeddings.append(embedding)
    return embeddings

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
    embeddings = embed_documents(split_texts)
    logger.info(f"Documents chunked. Sending to Milvus.")
    try:
        index = Milvus.from_texts(texts=split_texts, embedding=embeddings, connection_args=connection_info, collection_name=INDEX_NAME)
        return index
    except Exception as e:
        logger.error(f"Failed to index documents: {e}")
        raise

INDEXED = False

if __name__ == "__main__":
    logger.info("Starting the Milvus connection process")
    if INDEXED:
        index = connect(MILVUS_CONNECTION)
        logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    else:
        logger.info(f"Indexing at {MILVUS_CONNECTION}")
        index = index_documents(MILVUS_CONNECTION, SOURCE_FILE_NAMES, SOURCE_URLS, SOURCE_TITLES)

    query = "What is Data Mining?"
    try:
        results = index.similarity_search(query)
        print(results)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")

import os
import PyPDF2
import logging

#from langchain.vectorstores import Milvus
#from langchain.embeddings import HuggingFaceHubEmbeddings
#from langchain_community.embeddings import HuggingFaceHubEmbeddings
#from langchain_community.vectorstores import Milvus

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Informações dos arquivos para inserir na coleção (mdswift)
SOURCE_FILE_NAMES = ["Apostila_ML.pdf", "Artigo_ML.pdf"]
SOURCE_URLS = [
    "https://github.com/vieirasre/WatsonX_Assistant_with_Milvus/blob/main/Apostila_ML.pdf", 
    "https://github.com/vieirasre/WatsonX_Assistant_with_Milvus/blob/main/Artigo_ML.pdf"
]
SOURCE_TITLES = ["Apostila Machine Learning UFES", "Artigo Machine Learning UE"]
SOURCES_TOPIC = "Conteúdos Machine Learning"
INDEX_NAME = "ML_Collection4"

EMBED = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
MILVUS_HOST = os.environ.get("REMOTE_SERVER", '127.0.0.1')
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_CONNECTION = {"host": MILVUS_HOST, "port": MILVUS_PORT}

CHUNK_SIZE = 250
CHUNK_OVERLAP = 20

# Configuração do logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Logger initialized")

def connect(connection_info):
    logger.info("Connecting to Milvus with connection info: %s", connection_info)
    index = Milvus(
           EMBED,
           connection_args=connection_info,
           collection_name=INDEX_NAME,
           index_params="text"
       )
    logger.info("Connected to Milvus")
    return index

def index(connection_info, filenames, urls, titles):
    logger.info("Starting indexing process")
    texts, metadata = load_docs_pdf(filenames, urls, titles)
    logger.info("Loaded documents from PDFs")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_texts = text_splitter.create_documents(texts, metadata)
    logger.info("Documents chunked into smaller parts")
    
    logger.info("Sending chunked documents to Milvus")
    index = Milvus.from_documents(documents=split_texts, embedding=EMBED, connection_args=connection_info, collection_name=INDEX_NAME)
    logger.info("Indexing completed")
    
    return index

def load_docs_pdf(filenames, urls, titles):
    logger.info("Loading documents from PDFs")
    texts = []
    metadata = []
    i = 0
    for filename in filenames:
        logger.info("Processing file: %s", filename)
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
        logger.info("Finished processing file: %s", filename)
        i += 1
    logger.info("All documents loaded")
    return texts, metadata

INDEXED = False

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.info("Script started")
    if INDEXED:
        logger.info("Index exists. Connecting to existing index.")
        index = connect(MILVUS_CONNECTION)
    else:
        logger.info("No index found. Starting new indexing process.")
        index = index(MILVUS_CONNECTION, SOURCE_FILE_NAMES, SOURCE_URLS, SOURCE_TITLES)
    
    logger.info("Index setup completed. Ready for queries.")
    query = "What is Data Mining?"
    logger.info("Executing similarity search for query: %s", query)
    results = index.similarity_search(query)
    logger.info("Search results: %s", results)
    print(results)

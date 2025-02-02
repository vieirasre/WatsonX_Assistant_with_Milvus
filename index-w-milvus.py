import os, PyPDF2, logging

#from langchain.vectorstores import Milvus
#from langchain.embeddings import HuggingFaceHubEmbeddings
#from langchain_community.embeddings import HuggingFaceHubEmbeddings
#from langchain_community.vectorstores import Milvus

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_milvus import Milvus

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Informações dos arquivos para inserir na coleção (mdswift)
SOURCE_FILE_NAMES = ["Apostila_ML.pdf", "Artigo_ML.pdf"]
SOURCE_URLS = ["https://github.com/vieirasre/WatsonX_Assistant_with_Milvus/blob/main/Apostila_ML.pdf", 
               "https://github.com/vieirasre/WatsonX_Assistant_with_Milvus/blob/main/Artigo_ML.pdf"]
SOURCE_TITLES = ["Apostila Machine Learning UFES", "Artigo Machine Learning UE"]
SOURCES_TOPIC = "Conteúdos Machine Learning"
INDEX_NAME = "ML_Collection3"

EMBED = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
MILVUS_HOST = os.environ.get("REMOTE_SERVER", '127.0.0.1')
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_CONNECTION = {"host": MILVUS_HOST, "port": MILVUS_PORT}

CHUNK_SIZE = 250
CHUNK_OVERLAP = 20

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Logger initialized")

def connect(connection_info):
    index = Milvus(
           EMBED,
           connection_args=connection_info,
           collection_name=INDEX_NAME,
           index_params="text"
       )
    return index


def index(connection_info, filenames, urls, titles):
    texts, metadata = load_docs_pdf(filenames, urls, titles)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_texts = text_splitter.create_documents(texts, metadata)
    logging.info(f"Documents chunked.  Sending to Milvus.")
    index = Milvus.from_documents(documents=split_texts, embedding=EMBED, connection_args=connection_info, collection_name=INDEX_NAME)
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

INDEXED=False

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

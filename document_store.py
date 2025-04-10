import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

from config import CHROMADB_DIR, DOCUMENT_DIR, EMBEDDING_MODEL

vectorstore = None

def initialize_document_retrieval(document_directory=DOCUMENT_DIR):
    """Initialize the document retrieval system with the provided documents."""
    global vectorstore
    
    if os.path.exists(CHROMADB_DIR) and os.listdir(CHROMADB_DIR):
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=CHROMADB_DIR, embedding_function=embeddings)
        return vectorstore
    
    if not os.path.exists(document_directory):
        os.makedirs(document_directory)

    loader = DirectoryLoader(document_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMADB_DIR
    )
    vectorstore.persist()
    
    return vectorstore

def get_vectorstore():
    """Get the current vectorstore instance, initializing if necessary."""
    global vectorstore
    if vectorstore is None:
        vectorstore = initialize_document_retrieval()
    return vectorstore

def add_document(file_path):
    """Add a new document to the vectorstore."""
    global vectorstore
    
    if vectorstore is None:
        vectorstore = initialize_document_retrieval()
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    vectorstore.add_documents(chunks)
    vectorstore.persist()
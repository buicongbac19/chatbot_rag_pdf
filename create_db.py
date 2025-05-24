from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Khai bao bien
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def create_db_from_files():
    # Load all PDFs from the folder
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = text_splitter.split_documents(documents)
    print('Done chunking')
    # Set up Gemini API key
    os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY')  # Thay bằng API key của bạn

    # Use Gemini embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Create FAISS vector database (use from_documents, NOT from_texts)
    db = FAISS.from_documents(chunks, embedding_model)

    # Save database
    db.save_local(vector_db_path)
    return db

create_db_from_files()
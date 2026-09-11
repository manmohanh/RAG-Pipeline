from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "test.pdf"

#load the file into python program
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# split the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(documents=docs)

#Vector embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print('indexing of document done.')
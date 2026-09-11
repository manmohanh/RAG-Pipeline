from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_path = Path(__file__).parent / "test.pdf"

#load the file into python program
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

print(docs[2].page_content)
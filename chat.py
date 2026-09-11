from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
from google.genai import types


load_dotenv()

client = genai.Client()

#Vector embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

#take the user input
user_query = input("Ask your query: ")

#relevant chunks from vector db
search_results = vector_db.similarity_search(query=user_query)

context = "\n\n\n".join(
    [
    f"""Page Content: {result.page_content}
    Page Number: {result.metadata['page_label']}
    File Location: {result.metadata['source']}"""
        for result in search_results
    ]
)

SYSTEM_PROMPT= f"""
You are helpful AI assistant who answers user query based on the available  context 
retrieve from a PDF file along with page_content and page number.

You should only answer the user based on the following context and navigate the user
to open the right page number to know more.

Context:
{context}
"""


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_query,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT
    )
)

print(response.text)
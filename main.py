import os
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert OPENAI_API_KEY, "❌ OPENAI_API_KEY is missing from environment variables."


# --- Step 1: Extract Text from PDF ---
def extract_text_from_pdf(pdf_path):
    print(f"📄 Extracting text from {pdf_path}...")
    doc = fitz.open(pdf_path)
    return "".join([page.get_text() for page in doc])


# --- Step 2: Split Text into Chunks ---
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])


# --- Step 3: Embed Documents ---
def embed_documents(docs, embedding_model):
    print("🔍 Generating embeddings for document chunks...")
    return embedding_model.embed_documents([doc.page_content for doc in docs])


# --- Step 4: Store in Vector DB (Chroma) ---
def store_in_chroma(docs, embeddings, collection_name="springboot_docs"):
    print("📦 Storing embeddings in ChromaDB...")
    chroma_client = chromadb.Client(Settings())
    collection = chroma_client.get_or_create_collection(name=collection_name)
    for i, doc in enumerate(docs):
        collection.add(
            documents=[doc.page_content],
            ids=[f"doc_{i}"],
            embeddings=[embeddings[i]],
            metadatas=[{"source": "spring_boot_in_action"}]
        )
    return collection


# --- Step 5: Query Chroma and GPT ---
def query_rag(collection, embedding_model, query, api_key):
    print(f"❓ Querying: {query}")
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    context = " ".join(results["documents"][0])

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You're a helpful assistant that teaches Spring Boot."},
            {"role": "user", "content": f"Using this context:\n\n{context}\n\nAnswer this:\n{query}"}
        ]
    )
    return response.choices[0].message.content


# --- Main Pipeline ---
def main():
    # Step 0: Setup
    pdf_path = "spring_boot_in_action.pdf"
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Step 1–3
    text = extract_text_from_pdf(pdf_path)
    docs = chunk_text(text)
    embeddings = embed_documents(docs, embedding_model)

    # Step 4: Store
    collection = store_in_chroma(docs, embeddings)

    # Step 5: Query
    while True:
        question = input("\n❓ Ask me anything about Spring Boot (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        answer = query_rag(collection, embedding_model, question, OPENAI_API_KEY)
        print("\n💬 GPT Answer:\n", answer)


if __name__ == "__main__":
    main()

import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import openai
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index setup
dimension = 384
index_path = "faiss_index.bin"
document_chunks = []  # Flat list of chunks

# Load FAISS index if exists
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = faiss.IndexFlatL2(dimension)

# Folder for PDF files
STORAGE_DIR = "stored_pdfs"
os.makedirs(STORAGE_DIR, exist_ok=True)


# 🔹 Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


# 🔹 Chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# 🔹 Create vector store
def create_vector_store(chunks):
    embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


# 🔹 Store PDF embeddings
def store_pdf_embeddings(pdf_filename):
    global index, document_chunks

    pdf_path = os.path.join(STORAGE_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        return {"error": f"File '{pdf_filename}' not found in '{STORAGE_DIR}'."}

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Reset old values
    index, _ = create_vector_store(chunks)
    document_chunks = chunks

    # Save FAISS index
    faiss.write_index(index, index_path)

    print(f"✅ Stored {len(chunks)} chunks in FAISS.")
    return {"message": f"PDF '{pdf_filename}' stored successfully!"}


# 🔹 API to upload/store a PDF
@app.post("/store_pdf/")
async def store_pdf(pdf_filename: str):
    return store_pdf_embeddings(pdf_filename)


# 🔹 Retrieve most relevant chunk
def retrieve_relevant_chunk(query, top_k=3):
    if len(document_chunks) == 0:
        return "No PDF content loaded."

    query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    relevant_texts = []
    for idx in indices[0]:
        if idx < len(document_chunks):
            relevant_texts.append(document_chunks[idx])

    if not relevant_texts:
        return "No relevant content found."

    return " ".join(relevant_texts)


# 🔹 Generate response using OpenAI GPT
def generate_response(query, relevant_text):
    prompt = f"""
    You are an AI assistant. Based on the following extracted document content, answer the user's question concisely:

    DOCUMENT CONTEXT:
    {relevant_text}

    QUESTION: {query}

    Provide a brief summary.
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# 🔹 API to handle query from user
@app.get("/ask/")
async def ask_question(query: str):
    relevant_text = retrieve_relevant_chunk(query)
    if "No relevant content" in relevant_text:
        return {"response": relevant_text}

    return {"response": generate_response(query, relevant_text)}






# Souce venv/bin/activate


# Unicorn server:app —reload

# curl -X POST "http://127.0.0.1:8000/store_pdf/?pdf_filename=my_document.pdf"


# http://localhost:8501    - server url


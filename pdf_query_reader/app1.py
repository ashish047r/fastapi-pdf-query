# import streamlit as st
# import fitz  # PyMuPDF for extracting text
# import faiss
# import numpy as np
# import openai
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv

# import os



# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key=openai_api_key)

# # Initialize models
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)

# # Function to extract text from PDF
# def extract_text_from_pdf(uploaded_file):
#     text = ""

#     # Save the uploaded file temporarily
#     with open("temp_uploaded.pdf", "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Open the saved file using PyMuPDF
#     with fitz.open("temp_uploaded.pdf") as doc:
#         for page in doc:
#             text += page.get_text()

#     return text


# # Function to chunk text
# def chunk_text(text, chunk_size=500):
#     words = text.split()
#     chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
#     return chunks

# # Function to generate embeddings & store them in FAISS
# def create_vector_store(chunks):
#     embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks]).astype("float32")
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)
#     return index, embeddings, chunks

# # Function to retrieve relevant chunks using cosine similarity
# def retrieve_relevant_chunk(query, index, embeddings, chunks, top_k=3):
#     query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)
#     distances, indices = index.search(query_embedding, top_k)
#     return [chunks[idx] for idx in indices[0]]

# # Function to generate summarized response
# def generate_response(query, relevant_text):
#     prompt = f"""
#     You are an AI assistant. Based on the following extracted document content, answer the user's question concisely:

#     DOCUMENT CONTEXT:
#     {relevant_text}

#     QUESTION: {query}

#     Provide a brief summary.
#     """
#     response = llm.predict(prompt)
#     return response

# # Streamlit UI
# st.set_page_config(page_title="PDF Query Reader", page_icon="📄")
# st.title(" For Rheumatoid Arthritis")

# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# if uploaded_file is not None:
#     with st.spinner("Processing PDF..."):
#         text = extract_text_from_pdf(uploaded_file)
#         chunks = chunk_text(text)
#         index, embeddings, chunk_list = create_vector_store(chunks)
#         st.success("PDF processed successfully!")

#     query = st.text_input("Ask a question about the document:")

#     if query:
#         with st.spinner("Searching for answers..."):
#             relevant_chunks = retrieve_relevant_chunk(query, index, embeddings, chunk_list)
#             summarized_response = generate_response(query, " ".join(relevant_chunks))
#             st.write("### 🔍 Answer:")
#             st.write(summarized_response)







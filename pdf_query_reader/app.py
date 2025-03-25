import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000"

st.title("RA Query System")

# User Query Input (NO file upload required)
query = st.text_input("Ask a question about the stored document:")

if query:
    with st.spinner("Searching for the answer..."):
        response = requests.get(f"{API_URL}/ask/", params={"query": query})

        if response.status_code == 200:
            st.write("### 📝 Answer:")
            st.write(response.json()["response"])
        else:
            st.error("❌ Error fetching answer!")


# curl -X POST "http://127.0.0.1:8000/store_pdf/?pdf_filename=my_document.pdf"
# stremlit run app.py
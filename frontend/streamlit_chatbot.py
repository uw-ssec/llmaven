# streamlit_app.py

import streamlit as st
import requests, json
import sseclient

st.title("Custom RAG Chatbot (FastAPI Backend)")
st.markdown("This chatbot displays retrieved documents immediately, then updates with the generated answer.")

# Sidebar: file uploader and question input.
uploaded_files = st.sidebar.file_uploader("Attach PDF documents", type=["pdf"], accept_multiple_files=True)
user_question = st.text_input("Your question:")

# Function to convert uploaded files into payload documents.
def process_uploaded_files(files):
    documents = []
    if files:
        for file in files:
            # Read file content as text (for demonstration; use a proper parser in production)
            file_content = file.read().decode("utf-8", errors="ignore")
            documents.append({"filename": file.name, "content": file_content})
    return documents

if st.button("Submit Query") and user_question:
    payload = {
        "question": user_question,
        "documents": process_uploaded_files(uploaded_files)
    }
    # Replace with your API endpoint URL.
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
    response = requests.post(f"{API_URL}/api/query", json=payload, stream=True)
    
    client = sseclient.SSEClient(response)
    
    docs_displayed = False
    final_answer = ""
    
    for event in client.events():
        data = json.loads(event.data)
        if data["type"] == "docs" and not docs_displayed:
            st.subheader("Retrieved Document Chunks")
            for idx, chunk in enumerate(data["content"], start=1):
                st.write(f"**Chunk {idx}:** {chunk}")
            docs_displayed = True
        elif data["type"] == "final":
            final_answer = data["content"]
            st.subheader("Generated Response")
            st.write(final_answer)

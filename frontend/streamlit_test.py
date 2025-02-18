import streamlit as st
import time

def dummy_rag_response(question, documents):
    """Simulates a RAG response."""
    time.sleep(1)  # Simulate processing delay
    response = f"Dummy answer for: {question}"
    chunks = [
        "Dummy chunk 1: This is a simulated relevant snippet from an attached document.",
        "Dummy chunk 2: More simulated context related to your query."
    ]
    return {"response": response, "chunks": chunks}

st.title("Custom RAG Chatbot (Test Mode)")

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("chunks"):
            st.markdown("**Retrieved Document Chunks:**")
            for chunk in message["chunks"]:
                st.markdown(f"- {chunk}")

# File uploader for optional document attachments (PDFs)
uploaded_files = st.file_uploader("Attach documents (PDFs)", type=["pdf"], accept_multiple_files=True)

# Process uploaded files safely
documents = []
if uploaded_files:
    for file in uploaded_files:
        file_content = file.read()  # Read as binary to prevent decode errors
        documents.append({"filename": file.name, "content": "PDF content (binary data)"})  # Placeholder

# Chat input for the user question
if prompt := st.chat_input("Your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simulate processing with a loading spinner
    with st.spinner("Processing..."):
        result = dummy_rag_response(prompt, documents)

    assistant_message = {
        "role": "assistant",
        "content": result.get("response", ""),
        "chunks": result.get("chunks", [])
    }
    st.session_state.messages.append(assistant_message)

    with st.chat_message("assistant"):
        st.markdown(result.get("response", ""))
        st.markdown("### Retrieved Document Chunks:")
        for chunk in result.get("chunks", []):
            st.markdown(f"- {chunk}")

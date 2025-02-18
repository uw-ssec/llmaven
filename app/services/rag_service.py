# rag_application/app/services/rag_service.py

import textwrap
import asyncio
from typing import List, Optional, Tuple
from uuid import uuid4

# Import our modular core components.
from core.llms.download_model import load_transformer_model
from core.llms.llm_inference import generate_response
from core.retriever.vector_store import get_vector_store, create_vector_store_from_documents

class RAGService:
    def __init__(
        self,
        generation_model_config: Optional[dict] = None,
        embedding_model_config: Optional[dict] = None,
        vector_store_config: Optional[dict] = None,
    ):
        """
        Initialize the RAG service with configurable components.
        
        Parameters:
            generation_model_config (dict): Configuration to select the LLM and tokenizer.
            embedding_model_config (dict): Configuration for the embedding model.
            vector_store_config (dict): Configuration for the vector store.
        """
        # Load the generation model and tokenizer; these functions should use the provided config.
        self.model, self.tokenizer = load_transformer_model(config=generation_model_config)
        # Load (or initialize) the cached vector store using the provided config.
        self.cached_vector_store = get_vector_store(config=vector_store_config)
        
        # Define the base prompt template.
        self.input_prompt_template = textwrap.dedent(
            """\
            You are an astrophysics expert with a focus on the Rubin telescope project (formerly known as LSST).
            Please answer the question on astrophysics based on the following context:

            {context}

            Question: {question}
            """
        )
        self.k = 2  # Default number of document chunks to retrieve

    async def process_query(
        self, question: str, documents: Optional[List[dict]] = None
    ) -> Tuple[List[str], asyncio.Task]:
        """
        Retrieve document chunks and concurrently schedule the final answer generation.
        
        Parameters:
            question (str): The user query.
            documents (List[dict], optional): Documents to use for creating a new vector store.
                Each document should be a dict (e.g. {"filename": ..., "content": ...}).
        
        Returns:
            Tuple containing:
              - A list of retrieved document chunks (strings).
              - An asyncio.Task that will resolve to the final answer.
        """
        # Select vector store: use a new one if documents are provided, otherwise use the cached one.
        if documents:
            vector_store = create_vector_store_from_documents(documents)
        else:
            vector_store = self.cached_vector_store
        
        # Build the retriever from the chosen vector store.
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": self.k})
        
        # Optionally expand the query.
        expanded_query = self.expand_query_with_synonyms(question)
        # Retrieve document objects (each expected to have a 'page_content' attribute).
        docs_retrieved = retriever(expanded_query)
        retrieved_chunks = [doc.page_content for doc in docs_retrieved]
        
        # Schedule final answer generation concurrently.
        answer_task = asyncio.create_task(self.generate_answer(question, retrieved_chunks))
        return retrieved_chunks, answer_task

    async def generate_answer(self, question: str, retrieved_chunks: List[str]) -> str:
        """
        Construct the final prompt from retrieved document chunks and generate the final answer.
        
        Parameters:
            question (str): The original query.
            retrieved_chunks (List[str]): The text chunks retrieved from the documents.
        
        Returns:
            The generated answer as a string.
        """
        # Format the retrieved documents as context.
        context = "\n\n".join(retrieved_chunks)
        final_prompt = self.input_prompt_template.format(context=context, question=question)
        # Generate response using the asynchronous LLM inference.
        response = await generate_response(self.model, self.tokenizer, final_prompt)
        return response

    @staticmethod
    def expand_query_with_synonyms(query: str) -> str:
        """
        Optionally expand the query by appending synonyms or related keywords.
        """
        if "Rubin" in query:
            return query + " LSST Large Synoptic Survey Telescope"
        return query

# Optional test block to run the service standalone.
if __name__ == "__main__":
    import asyncio

    async def test():
        service = RAGService()
        test_question = "What are the latest discoveries from the Rubin telescope?"
        # For testing, we call process_query without additional documents.
        chunks, answer_task = await service.process_query(test_question, documents=None)
        print("Retrieved Chunks:")
        for chunk in chunks:
            print(chunk)
        answer = await answer_task
        print("\nGenerated Response:")
        print(answer)

    asyncio.run(test())

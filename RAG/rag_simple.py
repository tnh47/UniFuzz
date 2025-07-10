import os
import argparse
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SimpleRAG:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=self.api_key
        )
        self.vector_db_path = "./RAG/data/vector_db"

    def ask_simple(self, question):
        """Simple ask without LLM processing to avoid rate limits"""
        try:
            vector_store = FAISS.load_local(
                self.vector_db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            logging.info(f"Vector store loaded with {vector_store.index.ntotal} vectors")
        except Exception as e:
            logging.error(f"Error loading vector database: {e}")
            return "No relevant information available"

        try:
            # Simple similarity search - NO LLM calls
            docs = vector_store.similarity_search(question, k=3)
            if not docs:
                return "No relevant information available"
            
            # Return concatenated context
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                 for i, doc in enumerate(docs)])
            
            return f"Based on audit reports:\n\n{context}"
            
        except Exception as e:
            logging.error(f"Error in similarity search: {e}")
            return "No relevant information available"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple RAG without rate limits")
    parser.add_argument("command", choices=["ask"], help="Command: ask")
    parser.add_argument("--api-key", help="Google API Key")
    parser.add_argument("--question", help="Question for ask command")
    
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logging.error("API Key not found")
        exit(1)

    rag = SimpleRAG(api_key)
    
    if args.command == "ask":
        if not args.question:
            logging.error("Ask command requires --question parameter.")
            exit(1)
        result = rag.ask_simple(args.question)
        print("\nResponse:")
        print(result)

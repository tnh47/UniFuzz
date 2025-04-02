import os
import argparse
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RAG:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=self.api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4096,
            chunk_overlap=512
        )
        self.vector_db_path = "./RAG/data/vector_db"
    
    def ingest(self, pdf_dir="./RAG/data/audit_reports"):
        """Read PDF and store vector."""
        all_text = ""
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

        if not pdf_files:
            print(f"Cant find {pdf_dir}")
            return

        for filename in pdf_files:
            file_path = os.path.join(pdf_dir, filename)
            try:
                with open(file_path, "rb") as f:
                    pdf_reader = PdfReader(f)
                    for i, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                all_text += page_text + "\n"
                        except Exception as e:
                            logging.error(f"Error while read the page {i} in file {filename}: {e}")
            except Exception as e:
                logging.error(f"Open error{filename}: {e}")

        if not all_text:
            logging.error("Cant read anything from PDF!")
            return

        chunks = self.text_splitter.split_text(all_text)
        try:
            vector_store = FAISS.from_texts(chunks, self.embeddings)
            vector_store.save_local(self.vector_db_path)
            print(f"Vector were store with {len(chunks)} chunks.")
        except Exception as e:
            logging.error(f"Error while store vector: {e}")

    def ask(self, question):
        """Chat with RAG."""
        try:
            vector_store = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logging.error(f"Error while loading database{self.vector_db_path}: {e}")
            return

        try:
            docs = vector_store.similarity_search(question)
        except Exception as e:
            logging.error(f"Error while similarity searching: {e}")
            return

        prompt_template = (
            "Answer the question as detailed as possible from the provided context.\n"
            "Context:\n{context}\n\n"
            "{question}\n"
            "Answer:"
        )

        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            # model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=self.api_key
        )

        chain = load_qa_chain(
            model, 
            chain_type="stuff",
            prompt=PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        )

        try:
            response = chain({"input_documents": docs, "question": question})
            print("\n Respone from AI:")
            print(response["output_text"])
        except Exception as e:
            logging.error(f"Error query: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG chat by coolstar")
    parser.add_argument("command", choices=["ingest", "ask"], help="ingest or ask")
    parser.add_argument("--api-key", help="Google API Key")
    parser.add_argument("--question", help="question with ask")
    
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logging.error("API Key not found, use --api-key or enter in os env.")
        exit(1)

    rag = RAG(api_key)
    
    if args.command == "ingest":
        rag.ingest()
    elif args.command == "ask":
        if not args.question:
            logging.error("Ask command require --question.")
            exit(1)
        rag.ask(args.question)
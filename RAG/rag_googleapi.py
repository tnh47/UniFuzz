import os
import argparse
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import LLMChain
from func_timeout import func_timeout, FunctionTimedOut
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
    def extract_text_from_image(self, image):
        """Extract text from image using Tesseract OCR with error handling"""
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            logging.error(f"OCR Error: {e}")
            return ""
    def process_pdf_page(self, page, pdf_path, page_num):
        """Process PDF page combining text extraction and OCR for image-based pages"""
        text = page.extract_text() or ""
        
        # If text is empty or seems incomplete, try OCR
        if len(text.strip()) < 50:  # Threshold for considering as image page
            try:
                # Convert specific PDF page to image
                # Modified: Use tempfile to handle Windows path issues
                with tempfile.TemporaryDirectory() as temp_dir:
                    images = convert_from_path(
                        pdf_path,
                        first_page=page_num,
                        last_page=page_num,
                        output_folder=temp_dir,
                        fmt="jpeg"
                    )
                    
                    for img in images:
                        text += "\n" + self.extract_text_from_image(img)
            except Exception as e:
                logging.error(f"PDF to image conversion error: {e}")
        
        return text   
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
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            # PDF pages are 1-indexed in convert_from_path
                            processed_text = self.process_pdf_page(
                                page, 
                                file_path, 
                                page_num + 1  # convert_from_path uses 1-based page numbers
                            )
                            all_text += processed_text + "\n"
                        except Exception as e:
                            logging.error(f"Error processing page {page_num+1} in {filename}: {e}")
            except Exception as e:
                logging.error(f"Error opening {filename}: {e}")

        if not all_text.strip():
            logging.error("No text could be extracted from PDFs")
            return

        try:
            chunks = self.text_splitter.split_text(all_text)
            vector_store = FAISS.from_texts(chunks, self.embeddings)
            vector_store.save_local(self.vector_db_path)
            logging.info(f"Vector store updated with {len(chunks)} chunks")
        except Exception as e:
            logging.error(f"Vector store error: {e}")

    def ask(self, question):
        """Chat with RAG."""
        try:
            vector_store = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logging.error(f"Error while loading database{self.vector_db_path}: {e}")
            return
        compressor = LLMChainExtractor.from_llm(
            ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite",temperature=0, google_api_key=self.api_key)
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )
        decompose_template = """Phân tách "{question}" thành 3 câu hỏi con theo thứ tự:"""
        
        try:
            sub_questions = ChatGoogleGenerativeAI(
                temperature=0,
                google_api_key=self.api_key
            ).invoke(decompose_template.format(question=question)).content.split("\n")
        except Exception as e:
            logging.error(f"Lỗi phân tách câu hỏi: {e}")
            sub_questions = [question]

        all_docs = []
        for q in sub_questions:
            if q.strip():
                try:
                    docs = compression_retriever.get_relevant_documents(q.strip())
                    all_docs.extend(docs)
                except Exception as e:
                    logging.error(f"Lỗi tìm kiếm cho '{q}': {e}")

        prompt_template = (
            "Answer the question base on the provided context. If dont have enough information, just say 'Dont have matched information'\n"
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
            response = func_timeout(
                30, 
                chain.invoke, 
                args=({"input_documents": all_docs, "question": question},)
            )
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
import os
import argparse
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import cv2
import numpy as np
import base64
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="nomic-embed-text")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4096,
            chunk_overlap=512
        )
        self.vector_db_path = "./RAG/data/vector_db_ollama"
        self.llm = Ollama(model="llama3.2-vision", temperature=0.3)
        
    def _preprocess_image(self, image):
        """Tiền xử lý ảnh để cải thiện OCR"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
        return denoised

    def extract_text_from_image(self, image):
        """Kết hợp Tesseract và Llama Vision"""
        try:
            # Sử dụng Tesseract cho text đơn giản
            basic_text = pytesseract.image_to_string(image)
            
            # Sử dụng Llama Vision cho text phức tạp
            _, buffer = cv2.imencode('.png', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            vision_response = self.llm.generate({
                'model': 'llama3.2-vision',
                'prompt': 'Extract all text from this image',
                'images': [image_base64]
            })
            
            return f"{basic_text}\n[Llama Vision]\n{vision_response}"
        except Exception as e:
            logging.error(f"OCR Error: {e}")
            return ""

    def process_pdf_page(self, page, pdf_path, page_num):
        """Xử lý từng trang PDF với OCR nâng cao"""
        text = page.extract_text() or ""
        
        if len(text.strip()) < 50:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    images = convert_from_path(
                        pdf_path,
                        first_page=page_num,
                        last_page=page_num,
                        output_folder=temp_dir,
                        fmt="jpeg",
                        dpi=300
                    )
                    
                    for img in images:
                        processed_img = self._preprocess_image(np.array(img))
                        text += "\n" + self.extract_text_from_image(processed_img)
            except Exception as e:
                logging.error(f"PDF to image conversion error: {e}")
        
        return text

    def ingest(self, pdf_dir="./RAG/data/audit_reports2"):
        """Xử lý và lưu trữ tài liệu"""
        all_text = ""
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

        for filename in pdf_files:
            file_path = os.path.join(pdf_dir, filename)
            try:
                with open(file_path, "rb") as f:
                    pdf_reader = PdfReader(f)
                    for page_num, page in enumerate(pdf_reader.pages):
                        processed_text = self.process_pdf_page(
                            page, 
                            file_path, 
                            page_num + 1
                        )
                        all_text += processed_text + "\n"
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")

        chunks = self.text_splitter.split_text(all_text)
        vector_store = FAISS.from_texts(chunks, self.embeddings)
        vector_store.save_local(self.vector_db_path)
        logging.info(f"Đã lưu {len(chunks)} đoạn văn")

    def ask(self, question):
        """Xử lý truy vấn với RAG"""
        try:
            vector_store = FAISS.load_local(self.vector_db_path, self.embeddings)
            docs = vector_store.similarity_search(question, k=3)
            
            prompt_template = """
            [REQUEST]
            Answer base on context
            - markdown formart
            - list source information

            [CONTEXT]
            {context}

            [QUESTION]
            {question}
            """
            
            chain = load_qa_chain(
                self.llm,
                chain_type="stuff",
                prompt=PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
            )
            
            response = chain({"input_documents": docs, "question": question})
            print("\n[RESULT]")
            print(response["output_text"])
            
        except Exception as e:
            logging.error(f"Lỗi xử lý: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG với Ollama và OCR")
    parser.add_argument("command", choices=["ingest", "ask"])
    parser.add_argument("--question", help="Câu hỏi cần tra cứu")
    
    args = parser.parse_args()
    
    rag = RAG()
    
    if args.command == "ingest":
        rag.ingest()
    elif args.command == "ask":
        if not args.question:
            logging.error("Yêu cầu câu hỏi")
            exit(1)
        rag.ask(args.question)

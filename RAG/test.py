import os
import argparse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

class PDFChatCLI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4096,
            chunk_overlap=512
        )
    
    def ingest(self, pdf_dir="data/audit_reports"):
        """Xử lý và lưu trữ PDF"""
        text = ""
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                with open(os.path.join(pdf_dir, filename), "rb") as f:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
        
        chunks = self.text_splitter.split_text(text)
        vector_store = FAISS.from_texts(chunks, self.embeddings)
        vector_store.save_local("data/vector_db")
        print(f"✅ Đã xử lý {len(chunks)} chunks từ thư mục {pdf_dir}")

    def ask(self, question):
        """Hỏi đáp với dữ liệu đã xử lý"""
        vector_store = FAISS.load_local("data/vector_db", self.embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(question)
        
        prompt_template = """
        Answer the question as detailed as possible from the provided context.
        Context:\n{context}\n
        Question: {question}
        Answer:
        """
        
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
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
        
        response = chain({"input_documents": docs, "question": question})
        print("\n💡 Kết quả:")
        print(response["output_text"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Chat with PDFs")
    parser.add_argument("command", choices=["ingest", "ask"])
    parser.add_argument("--api-key", required=True, help="Google API Key")
    parser.add_argument("--question", help="Câu hỏi cần trả lời")
    
    args = parser.parse_args()
    
    chat = PDFChatCLI(args.api_key)
    
    if args.command == "ingest":
        chat.ingest()
    elif args.command == "ask" and args.question:
        chat.ask(args.question)
    else:
        print("Lệnh không hợp lệ!")
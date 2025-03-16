# -*- coding: utf-8 -*-
import os
import argparse
from pypdf import PdfReader  # Thêm dòng import này
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

# Cấu hình
EMBEDDING_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_DB_PATH = "vector_store"

def init_vector_db():
    """Khởi tạo vector database nếu chưa tồn tại"""
    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)
        # Tạo một vector store rỗng
        return FAISS.from_texts([""], HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    
    # Kiểm tra xem file index.faiss có tồn tại không
    if not os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        return FAISS.from_texts([""], HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    
    # Tải vector store từ thư mục
    return FAISS.load_local(
        VECTOR_DB_PATH,
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        allow_dangerous_deserialization=True
    )

def process_pdf(pdf_path):
    """Xử lý file PDF và trả về các chunks"""
    text = "\n".join([page.extract_text() for page in PdfReader(pdf_path).pages if page.extract_text()])
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def learn_command(pdf_path):
    """Xử lý lệnh học"""
    try:
        # Xử lý file/thư mục PDF
        if os.path.isdir(pdf_path):
            for file in os.listdir(pdf_path):
                if file.endswith(".pdf"):
                    chunks = process_pdf(os.path.join(pdf_path, file))
        else:
            chunks = process_pdf(pdf_path)
        
        # Cập nhật vector store
        vector_store = init_vector_db()
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        new_db = FAISS.from_texts(chunks, embeddings)
        vector_store.merge_from(new_db)
        vector_store.save_local(VECTOR_DB_PATH)
        
        # Tạo báo cáo học tập
        report_prompt = f"""
        Hãy tóm tắt những kiến thức đã học được từ các tài liệu sau:
        {chunks[:3]}... (trích dẫn 3 đoạn đầu)
        """
        response = ollama.generate(model="codellama", prompt=report_prompt)
        print("\n📚 Báo cáo học tập:")
        print(response["response"])
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")

def query_command():
    """Giao diện chat hỏi đáp"""
    vector_store = init_vector_db()
    print("\n💬 Chế độ hỏi đáp (Nhập 'exit' để thoát)")
    
    while True:
        query = input("\n🤔 Câu hỏi: ")
        if query.lower() == "exit":
            break
        
        # Truy xuất thông tin
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Tạo prompt
        prompt = f"""
        [VAI TRÒ] Bạn là chuyên gia bảo mật smart contract với kiến thức từ các audit report.
        [YÊU CẦU] Trả lời câu hỏi dựa trên ngữ cảnh và kiến thức chuyên môn.
        [NGỮ CẢNH] {context}
        [CÂU HỎI] {query}
        [TRẢ LỜI] Hãy trình bày chi tiết gồm:
        - Phân tích vấn đề
        - Ví dụ minh họa (nếu có)
        - Khuyến nghị giải pháp
        - Tham chiếu audit report liên quan
        """
        
        # Gọi LLM
        response = ollama.generate(model="codellama", prompt=prompt)
        print("\n💡 Trả lời:")
        print(response["response"])
        
        # Sinh test cases cho fuzzing
        fuzz_prompt = f"""
        Dựa trên câu trả lời trước, hãy đề xuất 3 test case cho công cụ fuzzing:
        {response["response"]}
        """
        fuzz_response = ollama.generate(model="codellama", prompt=fuzz_prompt)
        print("\n🔧 Đề xuất test cases cho fuzzing:")
        print(fuzz_response["response"])

def main():
    parser = argparse.ArgumentParser(description="Smart Contract Audit Expert System")
    subparsers = parser.add_subparsers(dest='command')
    
    # Lệnh học
    learn_parser = subparsers.add_parser('learn', help='Học từ file/thư mục PDF')
    learn_parser.add_argument('path', help='Đường dẫn file/thư mục PDF')
    
    # Lệnh hỏi
    subparsers.add_parser('query', help='Chế độ hỏi đáp')
    
    args = parser.parse_args()
    
    if args.command == "learn":
        learn_command(args.path)
    elif args.command == "query":
        query_command()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
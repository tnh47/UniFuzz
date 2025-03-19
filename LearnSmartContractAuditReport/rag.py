# -*- coding: utf-8 -*-
import os
import argparse
from pypdf import PdfReader
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
        return FAISS.from_texts([""], HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    return FAISS.load_local(
        VECTOR_DB_PATH,
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        allow_dangerous_deserialization=True
    )

def process_pdf(pdf_path):
    """Xử lý file PDF và trả về các chunks"""
    try:
        text = "\n".join([page.extract_text() for page in PdfReader(pdf_path).pages if page.extract_text()])
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_text(text)
    except Exception as e:
        raise ValueError(f"Lỗi xử lý file PDF: {str(e)}")

def learn_single_file(pdf_path):
    """Xử lý học từ 1 file PDF duy nhất"""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"File PDF không tồn tại: {pdf_path}")
    
    print(f"\n📖 Đang học từ file: {os.path.basename(pdf_path)}...")
    chunks = process_pdf(pdf_path)
    
    # Cập nhật vector store
    vector_store = init_vector_db()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    new_db = FAISS.from_texts(chunks, embeddings)
    vector_store.merge_from(new_db)
    vector_store.save_local(VECTOR_DB_PATH)
    
    # Báo cáo học tập
    report_prompt = f"""Hãy tóm tắt các điểm chính từ tài liệu:
    {chunks[:2]}... (trích dẫn 2 đoạn đầu)
    """
    response = ollama.generate(model="codellama", prompt=report_prompt)
    print("\n📚 Tóm tắt kiến thức đã học:")
    print(response["response"])
    print(f"\n✅ Đã học thành công từ file: {os.path.basename(pdf_path)}")
    print(f"🔢 Số lượng chunks đã thêm: {len(chunks)}")

def query_interface():
    """Giao diện hỏi đáp dựa trên vector database"""
    vector_store = init_vector_db()
    print("\n💬 Chế độ hỏi đáp dựa trên tri thức đã học (Nhập 'exit' để thoát)")
    
    while True:
        try:
            query = input("\n🎯 Câu hỏi của bạn: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            # Truy vấn vector database
            docs = vector_store.similarity_search(query, k=3)
            context = "\n".join([f"📄 Source {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
            
            # Tạo prompt
            prompt = f"""
            [VAI TRÒ] Bạn là chuyên gia smart contract với kiến thức từ các báo cáo audit
            [YÊU CẦU] Trả lời câu hỏi dựa HOÀN TOÀN vào thông tin sau:
            [NGỮ CẢNH]:
            {context}
            
            [CÂU HỎI]:
            {query}
            
            [TRẢ LỜI]:
            - Trình bày theo dạng điểm gạch đầu dòng
            - Kèm ví dụ code Solidity nếu có
            - Tham chiếu source tương ứng
            - Nếu không đủ thông tin hãy nói 'Không tìm thấy thông tin liên quan'
            """
            
            # Gọi LLM
            response = ollama.generate(
                model="codellama",
                prompt=prompt,
                options={"temperature": 0.3, "timeout": 60}
            )
            print("\n💡 Phân tích chuyên gia:")
            print(response["response"])
            
        except KeyboardInterrupt:
            print("\n🛑 Đã dừng chương trình")
            break
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Hệ thống Học và Phân tích Smart Contract")
    subparsers = parser.add_subparsers(dest='command')
    
    # Lệnh học từ file
    learn_parser = subparsers.add_parser('learn', help='Học từ một file PDF duy nhất')
    learn_parser.add_argument('pdf_file', help='Đường dẫn file PDF')
    
    # Lệnh hỏi đáp
    subparsers.add_parser('query', help='Chế độ hỏi đáp dựa trên tri thức đã học')
    
    args = parser.parse_args()
    
    try:
        if args.command == "learn":
            learn_single_file(args.pdf_file)
        elif args.command == "query":
            query_interface()
        else:
            parser.print_help()
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng: {str(e)}")

if __name__ == "__main__":
    main()
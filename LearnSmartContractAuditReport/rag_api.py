from pdfplumber import PDF
from langchain_text_splitters import TokenTextSplitter

def process_pdf(pdf_path: str) -> List[str]:
    """Xử lý PDF với pdfplumber và chia đoạn theo token"""
    try:
        print(f"🔍 Đang phân tích file: {os.path.basename(pdf_path)}")
        
        # Đọc PDF với pdfplumber
        text = ""
        with PDF.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"📄 Trang {i+1}:\n{page_text}\n\n"
                    else:
                        print(f"⚠️ Trang {i+1}: Không thể trích xuất văn bản")
                except Exception as e:
                    print(f"🔥 Lỗi trang {i+1}: {str(e)}")
        
        if not text.strip():
            raise ValueError("Không trích xuất được văn bản từ PDF")

        # Chia đoạn theo token với overlap
        splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            encoding_name="cl100k_base"
        )
        
        return splitter.split_text(text)
    
    except Exception as e:
        raise RuntimeError(f"PDF Processing Failed: {str(e)}")
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

class VectorDBManager:
    def __init__(self, embedding: Embeddings):
        self.db_path = "vector_db"
        self.embedding = embedding
        self.vector_store = self._init_db()
    
    def _init_db(self):
        """Khởi tạo FAISS database"""
        if os.path.exists(self.db_path):
            print("📂 Đang tải database từ local...")
            return FAISS.load_local(
                self.db_path,
                self.embedding,
                allow_dangerous_deserialization=True
            )
        return FAISS.from_texts([""], self.embedding)
    
    def update_db(self, chunks: List[str], metadata: dict):
        """Cập nhật database với dữ liệu mới"""
        new_db = FAISS.from_texts(
            texts=chunks,
            embedding=self.embedding,
            metadatas=[metadata]*len(chunks)
        )
        self.vector_store.merge_from(new_db)
        self.vector_store.save_local(self.db_path)
        print(f"💾 Đã lưu {len(chunks)} chunks vào database")

class AIMLEmbedder(Embeddings):
    """Custom Embedder cho AIML API"""
    def __init__(self):
        self.api_key = "6b8d5fc4e00546ee99f562af4e77c2d6"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed batch texts"""
        try:
            response = requests.post(
                "https://api.aimlapi.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "input": texts,
                    "model": "text-embedding-3-large",
                    "encoding_format": "float"
                },
                timeout=30
            )
            response.raise_for_status()
            return [item['embedding'] for item in response.json()['data']]
        except Exception as e:
            raise RuntimeError(f"Embedding Error: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        return self.embed_documents([text])[0]
def main_workflow(pdf_path: str):
    try:
        # 1. Xử lý PDF
        chunks = process_pdf(pdf_path)
        print(f"✅ Đã trích xuất {len(chunks)} đoạn văn bản")
        
        # 2. Khởi tạo hệ thống embedding
        embedder = AIMLEmbedder()
        db_manager = VectorDBManager(embedder)
        
        # 3. Cập nhật database
        metadata = {
            "source": os.path.basename(pdf_path),
            "timestamp": datetime.now().isoformat()
        }
        db_manager.update_db(chunks, metadata)
        
        print("🎉 Xử lý hoàn tất!")
    
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {str(e)}")
        # Ghi log lỗi chi tiết
        with open("processing_errors.log", "a") as f:
            f.write(f"[{datetime.now()}] {str(e)}\n")
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python rag_system.py <pdf_path>")
        sys.exit(1)
    
    main_workflow(sys.argv[1]
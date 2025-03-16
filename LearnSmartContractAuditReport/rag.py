# rag.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_vector_db(texts):
    """
    Tạo Vector Database từ danh sách văn bản.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    return FAISS.from_texts(texts, embeddings)

def retrieve_info(vector_db, query, k=3):
    """
    Truy xuất thông tin liên quan từ Vector Database.
    """
    return vector_db.similarity_search(query, k=k)
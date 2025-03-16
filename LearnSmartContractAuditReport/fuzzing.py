# fuzzing.py
from rag import create_vector_db, retrieve_info
from analyzer import analyze_smart_contract

def fuzz_smart_contract(pdf_path, ollama_model):
    """
    Phân tích và fuzz smart contract.
    """
    # Trích xuất văn bản từ PDF
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Tạo Vector Database
    vector_db = create_vector_db([text])

    # Phân tích smart contract
    guidance = analyze_smart_contract(text, ollama_model, vector_db)
    return guidance
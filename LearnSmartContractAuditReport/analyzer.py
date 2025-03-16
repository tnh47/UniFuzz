# analyzer.py
import ollama
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

def analyze_smart_contract(text, ollama_model, vector_db):
    """
    Phân tích smart contract dựa trên thông tin từ RAG.
    """
    try:
        prompt = f"""
        Based on the following smart contract audit report, extract:
        1. Common security vulnerabilities detected.
        2. Special testing conditions required.
        3. Critical logic or invariants that need verification.
        
        Audit report content:
        {text}
        """
        llm = Ollama(model=ollama_model)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        response = qa_chain.run(prompt)
        return response
    except Exception as e:
        return f"Error: {str(e)}"
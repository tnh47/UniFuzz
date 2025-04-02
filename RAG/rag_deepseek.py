from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re

# Cấu hình model
LLM_MODEL = "deepseek-r1:14b"
CONTEXT_WINDOW = 16384  # Độ dài context tối đa của model

# Load LLM với cấu hình tối ưu
def load_deepseek():
    return Ollama(
        model=LLM_MODEL,
        temperature=0.3,
        num_ctx=CONTEXT_WINDOW,
        top_k=30,
        top_p=0.9,
        system="Bạn là chuyên gia audit smart contract cấp cao với 15 năm kinh nghiệm."
    )

# Tạo prompt template cho smart audit
def create_audit_template():
    template = """<|im_start|>system
    [VAI TRÒ] Chuyên gia audit smart contract cấp cao
    [TRÁCH NHIỆM] Phân tích các rủi ro và đề xuất giải pháp
    
    [NGỮ CẢNH]:
    {context}
    
    [YÊU CẦU ĐẦU RA]:
    1. Phân tích từng điểm kỹ thuật
    2. Đánh giá mức độ nghiêm trọng (Low/Medium/High/Critical)
    3. Ví dụ code minh họa
    4. Tham chiếu nguồn tương ứng
    5. Kết luận tổng quan
    
    <|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant"""
    
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Xử lý response
def postprocess_response(response):
    cleaned = re.sub(r'<\|im_end\|>.*', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned

# Tạo chain cho smart audit
def create_audit_chain():
    llm = load_deepseek()
    prompt = create_audit_template()
    return LLMChain(llm=llm, prompt=prompt)

# Tích hợp với hệ thống hỏi đáp
def query_deepseek(question):
    chain = create_audit_chain()
    context = retrieve_context(question)  # Gia dinh co ham goi retrive context
    
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    return postprocess_response(response['text'])

# Cau hinh do sang tao
def set_dynamic_temperature(complexity):
    temperature_map = {
        "low": 0.1,
        "medium": 0.3,
        "high": 0.5
    }
    return temperature_map.get(complexity, 0.3)
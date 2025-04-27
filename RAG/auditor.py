# File: RAG/auditor.py
from typing import List, Dict
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class EnhancedAuditor:
    def __init__(self, rag_db_path, model_path):
        self.rag = AuditRAG()
        self.rag.load_vector_db(rag_db_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.ft_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.parser = JsonOutputParser(pydantic_object=AuditAnalysis)
        self.analysis_prompt = self._create_prompt_template()

    def _create_prompt_template(self):
        return PromptTemplate(
            template="""[INST] <<SYS>>
            You are a smart contract security expert. Analyze the contract and suggest fuzzing inputs.
            Context from previous audits: {audit_context}
            <</SYS>> {contract_code}
            {format_instructions} [/INST]""",
            input_variables=["contract_code", "audit_context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def analyze_contract(self, contract_code: str) -> Dict:
        audit_context = self.rag.query_relevant_findings(contract_code)
        inputs = self.tokenizer(
            self.analysis_prompt.format(
                contract_code=contract_code,
                audit_context=audit_context
            ),
            return_tensors="pt",
            max_length=4096,
            truncation=True
        ).to("cuda")
        
        outputs = self.ft_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class FuzzingRecommendation(BaseModel):
    """Structured fuzzing recommendations for detected vulnerabilities"""
    vulnerability_type: str = Field(..., description="Type of detected vulnerability")
    fuzzing_inputs: List[str] = Field(..., 
        description="Five concrete input examples to test this vulnerability",
        min_items=5,
        max_items=5
    )
    attack_scenario: str = Field(..., description="Brief attack scenario description")

class AuditAnalysis(BaseModel):
    """Structured audit analysis results"""
    detected_vulnerabilities: List[FuzzingRecommendation] = Field(
        ..., 
        description="List of detected vulnerabilities with fuzzing recommendations"
    )
    confidence_level: float = Field(
        ...,
        description="Model's confidence in the analysis (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

class SmartContractAuditor:
    """Analyzes smart contracts and generates fuzzing inputs"""
    
    def __init__(self, model_path: str = "codellama:7b"):
        self.llm = ChatOllama(
            model=model_path,
            temperature=0.1,
            num_ctx=4096,
            format="json",
            base_url="http://localhost:11434",
            timeout=60
        )
        self.parser = JsonOutputParser(pydantic_object=AuditAnalysis)
        
        self.analysis_prompt = PromptTemplate(
            template="""[INST]
            <<SYS>>
            You are a smart contract security expert. Analyze the provided contract code 
            and audit findings to generate fuzzing test inputs. Follow these rules:
            
            1. For each vulnerability, provide exactly 5 concrete fuzzing inputs
            2. Prioritize edge cases and invalid inputs
            3. Use real-world attack patterns
            4. Format inputs as they would appear in transaction calls
            
            Audit Context:
            {audit_context}
            <</SYS>>
            
            Target Contract Code:
            {contract_code}
            
            {format_instructions}
            [/INST]""",
            input_variables=["contract_code", "audit_context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def analyze_contract(self, contract_code: str, audit_context: str) -> Dict:
        """Analyze contract and generate fuzzing recommendations"""
        chain = self.analysis_prompt | self.llm | self.parser
        return chain.invoke({
            "contract_code": contract_code,
            "audit_context": audit_context
        })

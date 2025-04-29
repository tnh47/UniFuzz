#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
from typing import Dict, List, Any, Optional
try:
    # Thử import theo cách mới
    from google import generativeai as genai
except ImportError:
    # Fallback sang import theo cách cũ
    import google.generativeai as genai
from slither.slither import Slither
from fuzzer.engine.components.generator import Generator
from fuzzer.engine.components.individual import Individual

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AgentAnalyzer:
    def __init__(self, contract_path: str, api_key: str, solc_path: str = "/usr/bin/solc"):
        self.contract_path = contract_path
        self.api_key = api_key
        self.solc_path = solc_path
        self.analysis_result = None
        self.constructor_params = None
        self.function_inputs = {}
        
        # Cấu hình API theo cách tương thích với mọi phiên bản
        genai.configure(api_key=api_key)
        
    def analyze_contract(self) -> Dict:
        """Phân tích hợp đồng sử dụng Slither và LLM"""
        try:
            # Phân tích với Slither
            slither = Slither(self.contract_path, solc=self.solc_path)
            contract = slither.contracts[0]
            
            # Thu thập thông tin cơ bản
            contract_info = {
                "name": contract.name,
                "functions": [],
                "state_variables": [],
                "inheritance": [base.name for base in contract.inheritance],
                "constructor": None
            }
            
            # Phân tích constructor
            if contract.constructor:
                contract_info["constructor"] = {
                    "parameters": [
                        {
                            "name": param.name,
                            "type": str(param.type),
                            "visibility": param.visibility
                        }
                        for param in contract.constructor.parameters
                    ]
                }
            
            # Phân tích các hàm
            for func in contract.functions:
                if func.visibility in ["public", "external"]:
                    function_info = {
                        "name": func.name,
                        "parameters": [
                            {
                                "name": param.name,
                                "type": str(param.type),
                                "visibility": param.visibility
                            }
                            for param in func.parameters
                        ],
                        "visibility": func.visibility,
                        "state_mutability": func.state_mutability
                    }
                    contract_info["functions"].append(function_info)
            
            # Phân tích biến trạng thái
            for var in contract.state_variables:
                if var.visibility in ["public", "internal"]:
                    var_info = {
                        "name": var.name,
                        "type": str(var.type),
                        "visibility": var.visibility
                    }
                    contract_info["state_variables"].append(var_info)
            
            # Sử dụng LLM để phân tích sâu
            prompt = f"""
            Analyze this smart contract and identify potential security issues and interesting input values for fuzzing:
            
            Contract Info:
            {json.dumps(contract_info, indent=2)}
            
            Please provide:
            1. Critical functions that should be fuzzed
            2. Interesting input values for each function
            3. Potential security issues to focus on
            """
            
            # Gọi LLM API theo cách tương thích với mọi phiên bản
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            
            # Lấy text từ response (tương thích với các phiên bản khác nhau)
            response_text = response.text if hasattr(response, 'text') else response.parts[0].text
            
            # Lưu kết quả phân tích
            self.analysis_result = {
                "contract_info": contract_info,
                "llm_analysis": response_text
            }
            
            return self.analysis_result
            
        except Exception as e:
            logging.error(f"Error analyzing contract: {e}")
            return None

    def generate_constructor_params(self) -> Dict:
        """Sinh tham số constructor tối ưu"""
        if not self.analysis_result:
            self.analyze_contract()
            
        try:
            constructor_info = self.analysis_result["contract_info"]["constructor"]
            if not constructor_info:
                return {}
                
            prompt = f"""
            Generate optimal constructor parameters for fuzzing based on this constructor info:
            
            Constructor Parameters:
            {json.dumps(constructor_info["parameters"], indent=2)}
            
            Contract Analysis:
            {self.analysis_result["llm_analysis"]}
            
            Return a JSON object with parameter names as keys and their values.
            Focus on values that might trigger edge cases or security issues.
            """
            
            # Gọi LLM API theo cách tương thích với mọi phiên bản
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            
            # Lấy text từ response (tương thích với các phiên bản khác nhau)
            response_text = response.text if hasattr(response, 'text') else response.parts[0].text
            
            # Parse response và lưu constructor params
            self.constructor_params = json.loads(response_text)
            return self.constructor_params
            
        except Exception as e:
            logging.error(f"Error generating constructor params: {e}")
            return {}

    def generate_function_inputs(self, function_name: str, param_types: List[str]) -> List[Any]:
        """Sinh input tối ưu cho hàm cụ thể"""
        if not self.analysis_result:
            self.analyze_contract()
            
        try:
            # Tìm thông tin hàm
            function_info = None
            for func in self.analysis_result["contract_info"]["functions"]:
                if func["name"] == function_name:
                    function_info = func
                    break
                    
            if not function_info:
                return []
                
            prompt = f"""
            Generate optimal input values for fuzzing this function:
            
            Function: {function_name}
            Parameters: {json.dumps(function_info["parameters"], indent=2)}
            Parameter Types: {param_types}
            
            Contract Analysis:
            {self.analysis_result["llm_analysis"]}
            
            Return a list of values matching the parameter types.
            Focus on values that might trigger edge cases or security issues.
            """
            
            # Gọi LLM API theo cách tương thích với mọi phiên bản
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            
            # Lấy text từ response (tương thích với các phiên bản khác nhau)
            response_text = response.text if hasattr(response, 'text') else response.parts[0].text
            
            # Parse response và lưu function inputs
            inputs = json.loads(response_text)
            self.function_inputs[function_name] = inputs
            return inputs
            
        except Exception as e:
            logging.error(f"Error generating function inputs: {e}")
            return []

class AgentEnhancedGenerator(Generator):
    """Generator được tăng cường với agent phân tích"""
    
    def __init__(self, interface: Dict, bytecode: str, accounts: List[str], 
                 contract: str, agent_analyzer: Optional[AgentAnalyzer] = None, **kwargs):
        super().__init__(interface, bytecode, accounts, contract, **kwargs)
        self.agent_analyzer = agent_analyzer
        
    def generate_constructor(self) -> List[Dict]:
        """Sinh constructor với tham số tối ưu từ agent"""
        if self.agent_analyzer:
            constructor_params = self.agent_analyzer.generate_constructor_params()
            if constructor_params:
                return self._create_constructor_with_params(constructor_params)
        return super().generate_constructor()
        
    def _create_constructor_with_params(self, params: Dict) -> List[Dict]:
        """Tạo constructor với tham số cụ thể"""
        individual = []
        if "constructor" in self.interface and self.bytecode:
            arguments = ["constructor"]
            for param_name, param_value in params.items():
                arguments.append(param_value)
                
            individual.append({
                "account": self.get_random_account("constructor"),
                "contract": self.bytecode,
                "amount": self.get_random_amount("constructor"),
                "arguments": arguments,
                "blocknumber": self.get_random_blocknumber("constructor"),
                "timestamp": self.get_random_timestamp("constructor"),
                "gaslimit": self.get_random_gaslimit("constructor"),
                "returndatasize": {}
            })
        return individual
        
    def get_random_argument(self, type: str, function: str, argument_index: int) -> Any:
        """Sinh argument tối ưu từ agent"""
        if self.agent_analyzer:
            optimized_args = self.agent_analyzer.generate_function_inputs(function, [type])
            if optimized_args and len(optimized_args) > argument_index:
                return optimized_args[argument_index]
        return super().get_random_argument(type, function, argument_index)

def create_agent_enhanced_generator(contract_path: str, api_key: str, 
                                  interface: Dict, bytecode: str, 
                                  accounts: List[str], contract: str,
                                  solc_path: str = "/usr/bin/solc") -> AgentEnhancedGenerator:
    """Tạo generator được tăng cường với agent"""
    agent = AgentAnalyzer(contract_path, api_key, solc_path)
    return AgentEnhancedGenerator(
        interface=interface,
        bytecode=bytecode,
        accounts=accounts,
        contract=contract,
        agent_analyzer=agent
    )

def generate_test_cases_with_agent(generator: AgentEnhancedGenerator, num_cases: int = 100) -> List[Individual]:
    """Sinh test cases với generator được tăng cường"""
    test_cases = []
    for i in range(num_cases):
        test_case = generator.generate_random_individual()
        logging.info(f"[Agent] Test case {i+1}: {test_case}")
        test_cases.append(Individual(generator).init(chromosome=test_case))
    logging.info(f"[Agent] Tổng số test case được sinh ra: {len(test_cases)}")
    return test_cases

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
from typing import Dict, List, Any, Optional

from fuzzer.utils.utils import initialize_logger

class SmartContractAnalyzer:
    """
    Phân tích smart contract để xác định dataflow và mối quan hệ phụ thuộc giữa các hàm
    """
    def __init__(self, sol_path: str, api_key: str, solc_path: Optional[str] = None):
        self.sol_path = sol_path
        self.api_key = api_key
        self.solc_path = solc_path
        self.analysis_result = None
        self.dataflow_graph = None
        self.logger = initialize_logger("DataflowAnalyzer")
        
    def analyze(self):
        """Phân tích smart contract bằng Slither và LLM"""
        try:
            self.logger.info(f"Analyzing smart contract: {self.sol_path}")
            
            # Phân tích với Slither
            try:
                from slither.slither import Slither
                
                # Sử dụng solc_path nếu được cung cấp
                slither_args = {"solc": self.solc_path} if self.solc_path else {}
                slither = Slither(self.sol_path, **slither_args)
                
                # Xây dựng đồ thị phụ thuộc dữ liệu
                self.dataflow_graph = {}
                
                # Phân tích từng hợp đồng
                for contract in slither.contracts:
                    contract_info = {
                        "name": contract.name,
                        "functions": {},
                        "state_variables": [],
                        "inheritance": [base.name for base in contract.inheritance]
                    }
                    
                    # Thu thập biến trạng thái
                    for var in contract.state_variables:
                        var_info = {
                            "name": var.name,
                            "type": str(var.type),
                            "visibility": var.visibility
                        }
                        contract_info["state_variables"].append(var_info)
                    
                    # Thu thập thông tin các hàm
                    for func in contract.functions:
                        if func.visibility in ["public", "external"]:
                            # Bỏ qua hàm constructor vì được xử lý riêng
                            if func.name != "constructor":
                                function_info = {
                                    "name": func.name,
                                    "visibility": func.visibility,
                                    "state_mutability": getattr(func, "state_mutability", "nonpayable"),
                                    "reads": [v.name for v in func.state_variables_read],
                                    "writes": [v.name for v in func.state_variables_written],
                                    "calls": [],  # Sẽ được cập nhật bên dưới
                                    "parameters": [
                                        {"name": p.name, "type": str(p.type)}
                                        for p in func.parameters
                                    ]
                                }
                                
                                # Thu thập các lời gọi hàm nội bộ
                                for call in func.internal_calls:
                                    if hasattr(call, 'name'):
                                        function_info["calls"].append(call.name)
                                        
                                contract_info["functions"][func.name] = function_info
                    
                    self.dataflow_graph[contract.name] = contract_info
                
                # Chuyển đổi thành JSON để gửi đến LLM
                dataflow_json = json.dumps(self.dataflow_graph, indent=2)
                
                self.logger.info(f"Dataflow graph built successfully for {len(self.dataflow_graph)} contracts")
            except ImportError:
                self.logger.error("Không thể import Slither. Có thể thư viện này chưa được cài đặt.")
                self.logger.info("Tạo dataflow graph sơ bộ dựa trên đọc file...")
                
                # Đọc file Solidity và phân tích sơ bộ
                import re
                with open(self.sol_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Phân tích contracts
                contract_pattern = r'contract\s+(\w+)(?:\s+is\s+([^{]+))?\s*{([^}]+)}'
                contracts = re.findall(contract_pattern, code)
                
                self.dataflow_graph = {}
                
                for contract_match in contracts:
                    contract_name = contract_match[0].strip()
                    inheritance = [base.strip() for base in contract_match[1].split(',')] if contract_match[1] else []
                    contract_body = contract_match[2]
                    
                    # Phân tích state variables
                    var_pattern = r'(\w+(?:\[\])?\s+(?:public|private|internal)?\s+(\w+))'
                    state_vars = re.findall(var_pattern, contract_body)
                    
                    # Phân tích functions
                    func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*(public|external|internal|private)?\s*(view|pure|payable)?\s*(?:returns\s*\(([^)]*)\))?\s*{([^}]*)}'
                    functions = re.findall(func_pattern, contract_body)
                    
                    contract_info = {
                        "name": contract_name,
                        "functions": {},
                        "state_variables": [],
                        "inheritance": inheritance
                    }
                    
                    # Thêm state variables
                    for var in state_vars:
                        var_info = {
                            "name": var[1],
                            "type": var[0],
                            "visibility": "unknown"
                        }
                        contract_info["state_variables"].append(var_info)
                    
                    # Thêm functions
                    for func in functions:
                        func_name = func[0]
                        params_str = func[1]
                        visibility = func[2] if func[2] else "public"
                        mutability = func[3] if func[3] else "nonpayable"
                        
                        if visibility in ["public", "external"]:
                            params = []
                            if params_str:
                                param_list = params_str.split(',')
                                for p in param_list:
                                    parts = p.strip().split()
                                    if len(parts) >= 2:
                                        params.append({"name": parts[1], "type": parts[0]})
                            
                            function_info = {
                                "name": func_name,
                                "visibility": visibility,
                                "state_mutability": mutability,
                                "reads": [],
                                "writes": [],
                                "calls": [],
                                "parameters": params
                            }
                            
                            contract_info["functions"][func_name] = function_info
                    
                    self.dataflow_graph[contract_name] = contract_info
                
                # Chuyển đổi thành JSON để gửi đến LLM
                dataflow_json = json.dumps(self.dataflow_graph, indent=2)
                
                self.logger.info(f"Basic dataflow graph built from source code for {len(self.dataflow_graph)} contracts")
            
            # Gửi đến LLM để phân tích sâu hơn
            try:
                from google import generativeai as genai
                genai.configure(api_key=self.api_key)
                
                # Kiểm tra xem có thể bỏ qua LLM không (nếu đã vượt quota)
                skip_llm = False
                
                try:
                    # Thử gọi API đơn giản để kiểm tra quota
                    test_model = genai.GenerativeModel("gemini-1.5-pro")
                    test_response = test_model.generate_content("Hello, are you available?")
                    if not hasattr(test_response, 'text') and not hasattr(test_response, 'parts'):
                        skip_llm = True
                except Exception as quota_check_error:
                    self.logger.warning(f"LLM quota check failed, will skip LLM analysis: {str(quota_check_error)}")
                    skip_llm = True
                
                if skip_llm:
                    # Bỏ qua phân tích LLM, tạo kết quả mặc định
                    self.logger.warning("Skipping LLM analysis due to quota issues, using default analysis")
                    self.analysis_result = self._create_default_analysis()
                else:
                    # Giảm kích thước dữ liệu gửi đi
                    simplified_dataflow = {}
                    
                    # Chỉ lấy contract chính
                    main_contract_name = os.path.basename(self.sol_path).split('.')[0]
                    if main_contract_name in self.dataflow_graph:
                        # Chỉ lấy 5 functions quan trọng nhất
                        contract_info = self.dataflow_graph[main_contract_name]
                        functions = {}
                        
                        # Ưu tiên các functions có writes
                        write_funcs = {name: info for name, info in contract_info.get("functions", {}).items() 
                                      if info.get("writes")}
                        read_funcs = {name: info for name, info in contract_info.get("functions", {}).items() 
                                     if info.get("reads") and name not in write_funcs}
                        
                        # Lấy tối đa 3 hàm ghi và 2 hàm đọc
                        count = 0
                        for name, info in write_funcs.items():
                            if count < 3:
                                functions[name] = info
                                count += 1
                        
                        count = 0
                        for name, info in read_funcs.items():
                            if count < 2:
                                functions[name] = info
                                count += 1
                        
                        # Tạo contract info đơn giản
                        simplified_dataflow[main_contract_name] = {
                            "name": main_contract_name,
                            "functions": functions
                        }
                    
                    # Chuyển đổi thành JSON để gửi đến LLM
                    simplified_json = json.dumps(simplified_dataflow, indent=2)
                    
                    # Tạo prompt nhỏ gọn
                    minimal_prompt = f"""
                    Analyze this contract and suggest:
                    1. 2-3 critical test sequences
                    2. 1-2 potential vulnerabilities to check
                    
                    Contract: {main_contract_name}
                    Functions: {list(simplified_dataflow.get(main_contract_name, {}).get("functions", {}).keys())}
                    
                    Return ONLY a JSON with this format:
                    {{
                        "critical_paths": [["function1", "function2"], ...],
                        "test_sequences": [["function1", "function2"], ...],
                        "vulnerabilities": [{{
                            "type": "vulnerability_type",
                            "functions": ["function1"]
                        }}]
                    }}
                    """
                    
                    try:
                        self.logger.info("Sending minimal dataflow analysis to LLM")
                        model = genai.GenerativeModel("gemini-1.5-pro")
                        response = model.generate_content(minimal_prompt)
                        
                        # Lấy text từ response
                        response_text = response.text if hasattr(response, 'text') else response.parts[0].text
                        
                        # Parse kết quả JSON
                        self.analysis_result = json.loads(response_text)
                        self.logger.info(f"LLM analysis complete. Found {len(self.analysis_result.get('critical_paths', []))} critical paths, {len(self.analysis_result.get('test_sequences', []))} test sequences, and {len(self.analysis_result.get('vulnerabilities', []))} potential vulnerabilities")
                    except Exception as e:
                        self.logger.error(f"LLM analysis failed: {str(e)}")
                        # Tạo kết quả mặc định
                        self.analysis_result = self._create_default_analysis()
                
                # Lưu kết quả phân tích ra file cho việc gỡ lỗi
                with open("dataflow_analysis_result.json", "w") as f:
                    json.dump({
                        "dataflow_graph": simplified_dataflow if 'simplified_dataflow' in locals() else {},
                        "analysis_result": self.analysis_result
                    }, f, indent=2)
                
            except Exception as e:
                self.logger.error(f"Error during LLM analysis: {str(e)}")
                # Nếu LLM fails, tạo kết quả phân tích đơn giản
                self.analysis_result = self._create_default_analysis()
            
            return {
                "dataflow_graph": self.dataflow_graph,
                "analysis_result": self.analysis_result
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing smart contract: {str(e)}")
            # Trả về kết quả trống nếu có lỗi
            return {
                "dataflow_graph": {},
                "analysis_result": self._create_default_analysis()
            }
    
    def _create_default_analysis(self):
        """Tạo phân tích mặc định dựa trên tên hàm"""
        try:
            default_analysis = {
                "critical_paths": [],
                "test_sequences": [],
                "vulnerabilities": []
            }
            
            # Tìm các hàm từ dataflow graph
            functions = []
            if self.dataflow_graph:
                main_contract_name = os.path.basename(self.sol_path).split('.')[0]
                if main_contract_name in self.dataflow_graph:
                    functions = list(self.dataflow_graph[main_contract_name].get("functions", {}).keys())
            
            # Phân loại hàm dựa trên tên
            write_funcs = []
            read_funcs = []
            
            for func_name in functions:
                if func_name.lower().startswith(("set", "add", "create", "update", "delete", "remove", "transfer", "mint", "burn")):
                    write_funcs.append(func_name)
                elif func_name.lower().startswith(("get", "view", "is", "has", "balance", "total", "name", "symbol", "decimals")):
                    read_funcs.append(func_name)
            
            # Tạo test sequences đơn giản
            if write_funcs and read_funcs:
                # Tạo 2-3 sequences
                for i in range(min(3, len(write_funcs))):
                    write_func = write_funcs[i % len(write_funcs)]
                    read_func = read_funcs[i % len(read_funcs)]
                    default_analysis["test_sequences"].append([write_func, read_func])
                    default_analysis["critical_paths"].append([write_func, read_func])
            
            # Tạo các vulnerabilities đơn giản
            if "transfer" in functions or "transferFrom" in functions:
                default_analysis["vulnerabilities"].append({
                    "type": "reentrancy",
                    "functions": ["transfer"] if "transfer" in functions else ["transferFrom"]
                })
            
            if "approve" in functions:
                default_analysis["vulnerabilities"].append({
                    "type": "front-running",
                    "functions": ["approve"]
                })
            
            return default_analysis
        except Exception as e:
            self.logger.error(f"Error creating default analysis: {str(e)}")
            return {
                "critical_paths": [],
                "test_sequences": [],
                "vulnerabilities": []
            } 
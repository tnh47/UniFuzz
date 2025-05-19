#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import json
import logging
import os
import requests
import time
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from fuzzer.utils.utils import initialize_logger
from fuzzer.utils import settings
from .generator import Generator

class RAGEnhancedGenerator(Generator):
    """
    Generator sử dụng kết quả phân tích dataflow và RAG để sinh transaction sequence tối ưu
    """
    
    def __init__(self, interface: Dict, 
                 bytecode: str, 
                 accounts: List[str], 
                 contract: str, 
                 api_key: str,
                 analysis_result: Optional[Dict] = None,
                 contract_name: Optional[str] = None, 
                 sol_path: Optional[str] = None,
                 other_generators=None, 
                 interface_mapper=None):
        """
        Khởi tạo RAGEnhancedGenerator với các tham số bổ sung
        """
        super().__init__(interface, bytecode, accounts, contract, 
                        other_generators=other_generators, 
                        interface_mapper=interface_mapper,
                        contract_name=contract_name, 
                        sol_path=sol_path)
        
        self.api_key = api_key
        self.logger = initialize_logger("RAGEnhancedGenerator")
        
        # Lưu kết quả phân tích
        self.analysis_result = analysis_result or {
            "critical_paths": [],
            "test_sequences": [],
            "vulnerabilities": []
        }
        
        # Các sequence tối ưu từ phân tích
        self.optimal_sequences = self.analysis_result.get("test_sequences", [])
        
        # Vector lưu các critical path
        self.critical_paths = self.analysis_result.get("critical_paths", [])
        
        # Thông tin về lỗ hổng tiềm ẩn
        self.potential_vulnerabilities = self.analysis_result.get("vulnerabilities", [])
        
        # Cache cho các giá trị tham số
        self.arg_cache = {}
        
        # Danh sách các sequence tốt đã tìm thấy
        self.good_sequences = []
        
        # Cấu hình RAG server
        self.rag_api_endpoint = "http://localhost:5000/request"
        self.rag_timeout = 60  # seconds
        self.rag_max_retries = 3
        
        # Thống kê hiệu suất
        self.rag_requests = 0
        self.rag_successes = 0
        self.rag_failures = 0
        self.rag_cache_hits = 0
        
        self.logger.info(f"RAGEnhancedGenerator initialized with {len(self.optimal_sequences)} optimal sequences, {len(self.critical_paths)} critical paths, and {len(self.potential_vulnerabilities)} potential vulnerabilities")
    
    def _get_function_hash_by_name(self, function_name: str) -> Optional[str]:
        """Tìm function hash từ tên hàm"""
        if not self.interface_mapper:
            return None
            
        for fname, fhash in self.interface_mapper.items():
            # Chuẩn hóa tên hàm (loại bỏ phần tham số)
            normalized_name = fname.split("(")[0]
            if normalized_name == function_name:
                return fhash
        return None
    
    def _generate_optimal_sequence(self, sequence_template: List[str]) -> List[Dict]:
        """Sinh sequence dựa trên template từ phân tích dataflow"""
        sequence = []
        
        # Thêm constructor
        sequence.extend(self.generate_constructor())
        
        # Thêm các transaction theo template
        for func_name in sequence_template:
            func_hash = self._get_function_hash_by_name(func_name)
            if func_hash and func_hash in self.interface:
                # Sinh transaction với hàm được chỉ định
                transaction = self.generate_individual(func_hash, self.interface[func_hash])
                if transaction:  # Kiểm tra nếu generate_individual trả về transaction hợp lệ
                    sequence.extend(transaction)
        
        return sequence
    
    def _generate_vulnerability_targeting_sequence(self, vulnerability: Dict) -> List[Dict]:
        """Sinh sequence nhắm đến một lỗ hổng cụ thể"""
        sequence = []
        
        # Thêm constructor
        sequence.extend(self.generate_constructor())
        
        # Thêm các transaction liên quan đến lỗ hổng
        for func_name in vulnerability.get("functions", []):
            func_hash = self._get_function_hash_by_name(func_name)
            if func_hash and func_hash in self.interface:
                # Sinh transaction với hàm liên quan đến lỗ hổng
                transaction = self.generate_individual(func_hash, self.interface[func_hash], 
                                                     vuln_type=vulnerability.get("type"))
                if transaction:
                    sequence.extend(transaction)
        
        return sequence
    
    def _generate_related_functions_sequence(self, function_name: str) -> List[Dict]:
        """Sinh sequence với các hàm có liên quan đến hàm đã cho"""
        sequence = []
        
        # Thêm constructor
        sequence.extend(self.generate_constructor())
        
        # Tìm kiếm function hash từ tên
        target_func_hash = self._get_function_hash_by_name(function_name)
        if not target_func_hash:
            return sequence
        
        # Thêm target function transaction
        transaction = self.generate_individual(target_func_hash, self.interface[target_func_hash])
        if transaction:
            sequence.extend(transaction)
            
        # Tìm các hàm có liên quan trong critical paths
        related_functions = set()
        for path in self.critical_paths:
            if function_name in path:
                for func in path:
                    if func != function_name:
                        related_functions.add(func)
        
        # Thêm các transaction liên quan
        for rel_func_name in related_functions:
            rel_func_hash = self._get_function_hash_by_name(rel_func_name)
            if rel_func_hash and rel_func_hash in self.interface:
                rel_transaction = self.generate_individual(rel_func_hash, self.interface[rel_func_hash])
                if rel_transaction:
                    sequence.extend(rel_transaction)
        
        return sequence
    
    def _get_function_args_from_rag(self, function_name: str, function_hash: str, 
                                   argument_types: List[str], vuln_type: Optional[str] = None) -> Optional[List[Any]]:
        """Lấy tham số tối ưu từ RAG dựa trên dataflow và loại lỗi"""
        try:
            # Kiểm tra trong cache trước
            cache_key = f"{function_name}_{vuln_type}" if vuln_type else function_name
            if cache_key in self.arg_cache:
                self.rag_cache_hits += 1
                self.logger.info(f"Cache hit for {cache_key}")
                return self.arg_cache[cache_key]
            
            self.rag_requests += 1
                
            # Tìm lỗ hổng liên quan đến hàm này
            related_vulnerabilities = []
            for vuln in self.potential_vulnerabilities:
                if function_name in vuln.get("functions", []):
                    related_vulnerabilities.append(vuln)
            
            # Nếu có chỉ định loại lỗi, lọc chỉ lấy loại lỗi đó
            if vuln_type and related_vulnerabilities:
                related_vulnerabilities = [v for v in related_vulnerabilities if v.get("type") == vuln_type]
            
            # Xây dựng context từ dataflow
            dataflow_context = {}
            if hasattr(self, 'dataflow_graph'):
                for contract_name, contract_info in self.dataflow_graph.items():
                    if function_name in contract_info.get("functions", {}):
                        dataflow_context = contract_info["functions"][function_name]
                        break
            
            # Xây dựng prompt cải tiến cho RAG với ví dụ rõ ràng hơn
            prompt = f"""
Phân tích hợp đồng thông minh và tạo các giá trị tham số tối ưu cho fuzzing.

Contract: {self.contract_name}
Function: {function_name}
Parameter types: {argument_types}

CONTEXT PHÂN TÍCH:
{json.dumps(dataflow_context, indent=2) if dataflow_context else ""}

TÌM HIỂU CHỨC NĂNG HÀM: 
Hãy phân tích hàm {function_name} để hiểu:
- Chức năng chính của hàm này là gì?
- Các ràng buộc và điều kiện kiểm tra nào trong hàm?
- Các biến state nào bị ảnh hưởng bởi hàm này?
- Các hàm khác nào thường được gọi trước/sau hàm này?

PHÂN TÍCH ĐIỂM YẾU TIỀM ẨN:
Dựa trên các lỗ hổng phổ biến trong smart contract, xác định:
- Có khả năng xảy ra integer overflow/underflow không?
- Có ràng buộc access control nào có thể bị bypass không?
- Có thể xảy ra reentrancy không?
- Có vấn đề về logic trong điều kiện không?

SINH GIÁ TRỊ THAM SỐ:
Đối với mỗi tham số trong {argument_types}, hãy sinh giá trị đặc biệt dựa trên loại dữ liệu và chức năng của hàm:

- Đối với uint/int: Tìm giá trị biên, giá trị có thể bypass điều kiện, hoặc gây tràn số
- Đối với address: Tìm địa chỉ đặc biệt liên quan đến quyền hạn, tương tác với hợp đồng
- Đối với bool: Xác định giá trị có thể tác động đến control flow
- Đối với bytes/string: Xác định độ dài và nội dung có thể gây vấn đề

PHÂN TÍCH TRANSACTION SEQUENCE:
Dựa trên dataflow và critical paths, xác định:
- Các hàm nên được gọi trước {function_name}
- Các hàm nên được gọi sau {function_name}
- Trạng thái contract cần thiết trước khi gọi hàm này

CHỈ TRẢ VỀ:
Một mảng JSON đơn giản chỉ chứa các giá trị tham số, không có cấu trúc lồng nhau, không có tên trường. Ví dụ:
[
  "0x1234567890123456789012345678901234567890",  // địa chỉ
  1000000000  // số lượng
]

KHÔNG bao gồm giải thích hoặc metadata, chỉ trả về mảng JSON với các giá trị.
"""
            
            # Thêm thông tin về các lỗ hổng tiềm ẩn nếu có
            if related_vulnerabilities:
                prompt += f"""

THÔNG TIN LỖ HỔNG LIÊN QUAN:
{json.dumps(related_vulnerabilities, indent=2)}

Tập trung vào việc sinh các giá trị tham số có thể kích hoạt các lỗ hổng trên.
"""
            
            # Thêm thông tin về critical paths
            if self.critical_paths:
                prompt += f"""

CRITICAL PATHS:
{json.dumps(self.critical_paths, indent=2)}

Các paths này chỉ ra các chuỗi hàm có liên quan chặt chẽ với nhau. Sinh giá trị tham số phù hợp với các paths này.
"""
            
            # Thêm thông tin về các sequence tối ưu đã phát hiện
            if self.optimal_sequences:
                prompt += f"""

SEQUENCES TỐI ƯU ĐÃ PHÁT HIỆN:
{json.dumps(self.optimal_sequences, indent=2)}

Tham khảo các sequences tối ưu này khi sinh giá trị tham số.
"""
            
            # Thêm thông tin từ mã nguồn nếu có
            if hasattr(self, 'sol_path') and self.sol_path and os.path.exists(self.sol_path):
                try:
                    with open(self.sol_path, 'r') as f:
                        code = f.read()
                    # Tìm đoạn khai báo hàm cụ thể
                    function_pattern = re.compile(f"function\\s+{function_name}\\s*\\([^)]*\\)\\s*[^{{]*{{[^}}]*}}")
                    function_matches = function_pattern.findall(code)
                    if function_matches:
                        prompt += f"""

MÃ NGUỒN CỦA HÀM:
{function_matches[0]}

Phân tích mã nguồn để hiểu chính xác các ràng buộc, điều kiện và logic của hàm, từ đó sinh giá trị tham số phù hợp.
"""
                except Exception as e:
                    self.logger.warning(f"Could not read sol file: {e}")
            
            # Nhấn mạnh định dạng trả về để tránh lỗi JSON
            prompt += """
QUAN TRỌNG: Trả về mảng JSON đơn giản, ví dụ:
[
  "0x1234567890123456789012345678901234567890",
  1000000000
]

KHÔNG bao gồm cấu trúc phức tạp hoặc metadata như "parameter_name", "description", "function", "parameters",...
CHỈ trả về một mảng chứa các giá trị theo thứ tự tham số.
"""
            
            self.logger.info(f"Requesting argument values from RAG for {function_name}")
            
            # Gọi RAG API
            rag_response = self._fetch_rag_suggestion(prompt)
            
            if not rag_response:
                self.rag_failures += 1
                self.logger.warning(f"RAG returned no response for {function_name}")
                return None
                
            # Xử lý phản hồi
            try:
                # Làm sạch chuỗi JSON - xóa các format code blocks và comments
                clean_response = rag_response.strip()
                
                # Loại bỏ code blocks
                clean_response = re.sub(r"```json\s*|\s*```", "", clean_response)
                
                # Loại bỏ comments
                clean_response = re.sub(r"//.*?$", "", clean_response, flags=re.MULTILINE)
                clean_response = re.sub(r"/\*.*?\*/", "", clean_response, flags=re.DOTALL)
                
                # Tìm mảng JSON trong phản hồi
                json_array_match = re.search(r"\[\s*.*?\s*\]", clean_response, re.DOTALL)
                if json_array_match:
                    clean_response = json_array_match.group(0)
                
                # Xử lý chuỗi đặc biệt như 2**256-1
                clean_response = re.sub(r'(\d+)\s*\*\*\s*(\d+)\s*-\s*(\d+)', lambda m: str((int(m.group(1)) ** int(m.group(2))) - int(m.group(3))), clean_response)
                clean_response = re.sub(r'(\d+)\s*\*\*\s*(\d+)', lambda m: str(int(m.group(1)) ** int(m.group(2))), clean_response)
                
                # Thử parse JSON
                args = None
                try:
                    args = json.loads(clean_response)
                except json.JSONDecodeError:
                    # Nếu không parse được, thử trích xuất các giá trị theo mẫu dựa trên kiểu dữ liệu
                    self.logger.warning(f"Could not parse as JSON, trying to extract values manually: {clean_response}")
                    
                    # Sinh giá trị mặc định dựa trên kiểu tham số
                    args = []
                    for arg_type in argument_types:
                        args.append(self._get_interesting_value_for_type(arg_type))
                    
                # Kiểm tra xem args có phải là cấu trúc lồng nhau không
                if isinstance(args, list) and any(isinstance(item, dict) for item in args):
                    self.logger.warning(f"RAG returned nested structure, extracting values")
                    
                    # Trích xuất giá trị từ cấu trúc lồng nhau
                    processed_args = []
                    for i, arg_type in enumerate(argument_types):
                        if i < len(args):
                            # Trích xuất giá trị dựa trên cấu trúc
                            item = args[i]
                            if isinstance(item, dict):
                                # Tìm kiếm giá trị trong các trường phổ biến
                                if "value" in item:
                                    processed_args.append(item["value"])
                                elif "parameters" in item and isinstance(item["parameters"], dict):
                                    # Lấy giá trị đầu tiên từ parameters
                                    param_values = list(item["parameters"].values())
                                    if param_values:
                                        processed_args.append(param_values[0])
                                    else:
                                        processed_args.append(self._get_interesting_value_for_type(arg_type))
                                else:
                                    # Lấy giá trị đầu tiên từ dict
                                    values = list(item.values())
                                    if values:
                                        processed_args.append(values[0])
                                    else:
                                        processed_args.append(self._get_interesting_value_for_type(arg_type))
                            else:
                                processed_args.append(item)
                        else:
                            processed_args.append(self._get_interesting_value_for_type(arg_type))
                    
                    args = processed_args
                
                # Đảm bảo args là list
                if not isinstance(args, list):
                    self.logger.warning(f"Expected list response, got {type(args)}")
                    args = [args]
                
                # Đảm bảo đủ số lượng tham số
                while len(args) < len(argument_types):
                    args.append(self._get_interesting_value_for_type(argument_types[len(args)]))
                
                # Chuyển đổi các giá trị trong args sang định dạng hợp lệ
                processed_args = []
                for i, arg in enumerate(args):
                    if i < len(argument_types):
                        # Xử lý dựa trên kiểu dữ liệu mong đợi
                        if argument_types[i].startswith(("uint", "int")):
                            # Xử lý số nguyên
                            if isinstance(arg, str):
                                try:
                                    if arg.startswith("0x"):
                                        processed_args.append(int(arg, 16))
                                    else:
                                        # Xử lý các biểu thức như 2**256-1 (nếu còn sót)
                                        if "**" in arg or "-" in arg or "+" in arg or "*" in arg:
                                            try:
                                                processed_args.append(eval(arg))
                                            except:
                                                processed_args.append(int(arg) if arg.isdigit() else 0)
                                        else:
                                            processed_args.append(int(arg) if arg.isdigit() else 0)
                                except ValueError:
                                    processed_args.append(0)
                            else:
                                processed_args.append(arg if isinstance(arg, int) else 0)
                        elif argument_types[i] == "address":
                            # Xử lý địa chỉ
                            if isinstance(arg, str) and arg.startswith("0x") and len(arg) == 42:
                                processed_args.append(arg)
                            else:
                                # Nếu không phải địa chỉ hợp lệ, dùng địa chỉ từ accounts
                                if len(self.accounts) > 0:
                                    processed_args.append(random.choice(self.accounts))
                                else:
                                    processed_args.append("0x0000000000000000000000000000000000000000")
                        else:
                            processed_args.append(arg)
                
                self.logger.info(f"RAG suggested args for {function_name}: {processed_args}")
                
                # Lưu vào cache
                self.arg_cache[cache_key] = processed_args
                self.rag_successes += 1
                
                return processed_args
                
            except Exception as e:
                self.rag_failures += 1
                self.logger.error(f"JSON parse error for {function_name}: {e}")
                self.logger.error(f"RAG response: {rag_response}")
                # Trả về giá trị mặc định
                return [self._get_interesting_value_for_type(arg_type) for arg_type in argument_types]
            
        except Exception as e:
            self.rag_failures += 1
            self.logger.error(f"Error getting function args from RAG: {e}")
            return None
    
    def _get_interesting_value_for_type(self, type_str: str) -> Any:
        """
        Trả về một giá trị thú vị (không phải mặc định) cho một kiểu dữ liệu để cải thiện fuzzing
        """
        if type_str.startswith("uint"):
            # Trả về một giá trị thú vị cho uint
            interesting_values = [
                0,                    # Zero
                1,                    # One
                2**256 - 1,           # MAX_UINT
                2**128 - 1,           # Big number
                2**64 - 1,            # Another big number
                1000000,              # Medium number
                10                    # Small number
            ]
            return random.choice(interesting_values)
        elif type_str.startswith("int"):
            # Trả về một giá trị thú vị cho int
            interesting_values = [
                0,                    # Zero
                1,                    # One
                -1,                   # Negative one
                2**127 - 1,           # MAX_INT
                -(2**127),            # MIN_INT
                1000000,              # Medium positive
                -1000000              # Medium negative
            ]
            return random.choice(interesting_values)
        elif type_str == "address":
            # Trả về một địa chỉ thú vị
            interesting_addresses = [
                "0x0000000000000000000000000000000000000000",  # Zero address
                "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",  # Max address
                "0x1000000000000000000000000000000000000000",  # Special address 1
                "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"   # A common address used in DeFi
            ]
            return random.choice(interesting_addresses)
        elif type_str == "bool":
            return random.choice([True, False])
        elif type_str.startswith("bytes"):
            interesting_bytes = [
                "0x00",                               # Empty bytes
                "0xFFFFFFFF",                         # All Fs
                "0x1234567890ABCDEF",                 # Random hex
                "0x" + "00" * 32                      # Long zeros
            ]
            return random.choice(interesting_bytes)
        elif type_str == "string":
            interesting_strings = [
                "",                                  # Empty string
                "Hello",                             # Normal string
                "A" * 100,                           # Long string
                "Special@#$%^&*()Characters"         # Special characters
            ]
            return random.choice(interesting_strings)
        else:
            return None
    
    def _get_default_value_for_type(self, type_str: str) -> Any:
        """Trả về giá trị mặc định cho một kiểu dữ liệu"""
        if type_str.startswith("uint") or type_str.startswith("int"):
            return 0
        elif type_str == "address":
            return "0x0000000000000000000000000000000000000000"
        elif type_str == "bool":
            return False
        elif type_str.startswith("bytes"):
            return "0x00"
        elif type_str == "string":
            return ""
        else:
            return None
    
    def generate_individual(self, function: str, argument_types: List[str], 
                           vuln_type: Optional[str] = None, default_value: bool = False) -> List[Dict]:
        """
        Sinh transaction dùng sequence thông minh từ RAG nhưng giá trị từ generator gốc
        để tối ưu độ phủ code
        """
        # Sử dụng trực tiếp hàm gốc từ lớp cha để tạo ra các giá trị tham số
        # Điều này giúp tăng cường độ phủ code và đảm bảo tính đúng đắn của tham số
        return super().generate_individual(function, argument_types, default_value)
    
    def _get_suggested_sequence_from_rag(self) -> Optional[List[str]]:
        """Lấy gợi ý sequence từ RAG."""
        prompt = f"""
Dựa vào smart contract '{self.contract_name}' với các hàm: {list(self.interface_mapper.keys()) if self.interface_mapper else list(self.interface.keys())},
hãy đề xuất một chuỗi các lời gọi hàm để tìm lỗ hổng tiềm ẩn hoặc tăng độ phủ code.

Thông tin hợp đồng: {self.contract_name}
Các hàm có sẵn: {', '.join(list(self.interface_mapper.keys()) if self.interface_mapper else list(self.interface.keys()))}

Xem xét critical paths: {json.dumps(self.critical_paths, indent=2)} 
Và các lỗ hổng tiềm ẩn: {json.dumps(self.potential_vulnerabilities, indent=2)}.

Một sequence tốt sẽ:
1. Thay đổi trạng thái contract theo cách có thể dẫn đến lỗi
2. Kiểm tra các điều kiện biên và giá trị đặc biệt
3. Tương tác với các hàm có quan hệ phụ thuộc dữ liệu
4. Nhắm vào các lỗ hổng tiềm ẩn

Chỉ trả về một mảng JSON chứa tên các hàm (ví dụ: ["transfer", "approve", "transferFrom"])
"""
        response = self._fetch_rag_suggestion(prompt)
        if response:
            try:
                clean_response = re.sub(r"```json\s*|\s*```", "", response).strip()
                json_array_match = re.search(r"\[\s*.*?\s*\]", clean_response, re.DOTALL)
                if json_array_match:
                    clean_response = json_array_match.group(0)
                sequence = json.loads(clean_response)
                if isinstance(sequence, list) and all(isinstance(f, str) for f in sequence):
                    return sequence
            except Exception as e:
                self.logger.warning(f"Không thể parse sequence từ RAG: {e}")
        return None

    def generate_random_individual(self, func_hash=None, func_args_types=None, default_value=False):
        """
        Sinh chuỗi transaction thông minh dựa trên phân tích dataflow và lịch sử của các chuỗi thành công
        nhưng giữ nguyên cách sinh giá trị từ generator gốc để đảm bảo độ phủ code cao
        """
        # Nếu đã chỉ định hash và args, dùng chúng
        if func_hash is not None and func_args_types is not None:
            individual = []
            individual.extend(self.generate_constructor())
            # Sử dụng phương thức từ generator gốc để sinh giá trị
            individual.extend(super().generate_individual(func_hash, func_args_types, default_value=default_value))
            return individual
        
        # Khởi tạo chuỗi giao dịch với constructor
        individual = []
        individual.extend(self.generate_constructor())
        
        # Đếm số lượng hàm sinh từ generator gốc vs từ RAG để log thông tin
        original_functions = 0
        rag_enhanced_functions = 0
        
        # Tạo danh sách các hàm để sử dụng cho sequence để tránh lặp lại
        available_functions = list(self.interface.keys())
        random.shuffle(available_functions)
        
        # Quyết định cách sinh sequence dựa vào chiến lược
        # Điều chỉnh weights: tránh lặp lại optimal_sequence và critical_path liên tục
        # Tăng tỷ lệ sử dụng mutation và combined để đa dạng hóa sequence
        strategy = random.choices(
            ["optimal_sequence", "critical_path", "mutation", "random", "combined"],
            weights=[0.25, 0.25, 0.2, 0.1, 0.2]
        )[0]
        
        # Đếm số lần sử dụng mỗi chiến lược
        if not hasattr(self, '_strategy_counts'):
            self._strategy_counts = {
                "optimal_sequence": 0, 
                "critical_path": 0, 
                "mutation": 0, 
                "random": 0, 
                "combined": 0
            }
        self._strategy_counts[strategy] += 1
        
        self.logger.info(f"Using strategy: {strategy} for transaction sequence")

        # Giới hạn số lượng transaction tối đa
        MAX_SEQUENCE_LENGTH = 5  # Đặt số lượng transaction tối đa nhỏ hơn để tối ưu độ phủ

        # Đảm bảo độ đa dạng bằng cách không lặp lại sequence từ optimal/critical path
        if (strategy == "optimal_sequence" or strategy == "critical_path") and hasattr(self, '_last_sequence'):
            # Nếu strategy trước đó cũng là optimal hoặc critical và đã sử dụng > 5 lần liên tiếp
            # thì chuyển sang chiến lược khác để tăng độ đa dạng
            if self._last_sequence == strategy:
                self._repeat_count = getattr(self, '_repeat_count', 0) + 1
                if self._repeat_count > 5:
                    strategy = random.choice(["mutation", "combined", "random"])
                    self.logger.info(f"Switching to {strategy} to increase diversity")
                    self._repeat_count = 0
            else:
                self._repeat_count = 1
        
        self._last_sequence = strategy

        # Set random seed dựa trên thời gian để tăng tính ngẫu nhiên
        random.seed(time.time() + random.random())

        if strategy == "optimal_sequence" and self.optimal_sequences:
            # Sử dụng sequence từ phân tích dataflow
            sequence_template = random.choice(self.optimal_sequences)
            self.logger.info(f"Using optimal sequence template: {sequence_template}")
            
            # Giới hạn số lượng transaction trong template
            sequence_template = sequence_template[:MAX_SEQUENCE_LENGTH-1]  # Để lại chỗ cho 1 hàm ngẫu nhiên
            
            # Sinh transaction theo template
            for func_name in sequence_template:
                func_hash = self._get_function_hash_by_name(func_name)
                if func_hash and func_hash in self.interface:
                    # Sinh giá trị từ generator gốc để đảm bảo phủ code tốt
                    tx = super().generate_individual(func_hash, self.interface[func_hash], default_value=default_value)
                    if tx:
                        individual.extend(tx)
                        rag_enhanced_functions += 1
            
            # Thêm một hàm ngẫu nhiên không có trong template để tăng đa dạng
            if len(available_functions) > 0 and len(individual) < MAX_SEQUENCE_LENGTH + 1:  # +1 cho constructor
                # Lọc ra các hàm không có trong template
                used_hashes = set()
                for tx in individual[1:]:  # Bỏ qua constructor
                    if "arguments" in tx and tx["arguments"]:
                        used_hashes.add(tx["arguments"][0])
                
                unused_functions = [f for f in available_functions if f not in used_hashes]
                if unused_functions:
                    random_func = random.choice(unused_functions)
                    tx = super().generate_individual(random_func, self.interface[random_func], default_value=default_value)
                    if tx:
                        individual.extend(tx)
                        original_functions += 1
                        self.logger.info(f"Added random function {random_func[:8]} for diversity")
        
        elif strategy == "critical_path" and self.critical_paths:
            # Sử dụng critical path từ phân tích dataflow
            path = random.choice(self.critical_paths)
            self.logger.info(f"Following critical path: {path}")
            
            # Giới hạn số lượng transaction trong path
            path = path[:MAX_SEQUENCE_LENGTH-1]  # Để lại chỗ cho 1 hàm ngẫu nhiên
            
            for func_name in path:
                func_hash = self._get_function_hash_by_name(func_name)
                if func_hash and func_hash in self.interface:
                    # Sinh giá trị từ generator gốc để đảm bảo phủ code tốt
                    tx = super().generate_individual(func_hash, self.interface[func_hash], default_value=default_value)
                    if tx:
                        individual.extend(tx)
                        rag_enhanced_functions += 1
            
            # Thêm một hàm ngẫu nhiên không có trong path để tăng đa dạng
            if len(available_functions) > 0 and len(individual) < MAX_SEQUENCE_LENGTH + 1:  # +1 cho constructor
                # Lọc ra các hàm không có trong path
                used_hashes = set()
                for tx in individual[1:]:  # Bỏ qua constructor
                    if "arguments" in tx and tx["arguments"]:
                        used_hashes.add(tx["arguments"][0])
                
                unused_functions = [f for f in available_functions if f not in used_hashes]
                if unused_functions:
                    random_func = random.choice(unused_functions)
                    tx = super().generate_individual(random_func, self.interface[random_func], default_value=default_value)
                    if tx:
                        individual.extend(tx)
                        original_functions += 1
                        self.logger.info(f"Added random function {random_func[:8]} for diversity")
        
        elif strategy == "mutation" and hasattr(self, 'population') and self.population:
            # Đột biến một sequence tốt từ quần thể
            if len(self.population) > 0:
                # Chọn một sequence tốt từ quần thể
                # Ưu tiên chọn từ top 5 nhưng đôi khi cũng chọn từ các cá thể xa hơn để đa dạng hóa
                max_idx = len(self.population) - 1
                if max_idx > 10 and random.random() < 0.3:  # 30% thời gian, lấy từ cá thể xa hơn
                    best_idx = random.randint(5, min(10, max_idx))
                else:
                    best_idx = random.randint(0, min(5, max_idx))
                
                if hasattr(self.population[best_idx], "individual"):
                    base_sequence = self.population[best_idx]["individual"]
                elif hasattr(self.population[best_idx], "chromosome"):
                    base_sequence = self.population[best_idx].chromosome
                else:
                    base_sequence = self.population[best_idx]
                
                if base_sequence and len(base_sequence) > 1:  # Bỏ qua constructor
                    # Áp dụng thuật toán đột biến hoàn chỉnh từ hàm mutate() nhưng với xác suất tùy chỉnh
                    mutated_indiv = self.mutate(base_sequence)
                    self.logger.info(f"Applied full mutation to sequence from population index {best_idx}")
                    return mutated_indiv
        
        elif strategy == "combined":
            # Kết hợp từ nhiều nguồn để đa dạng hóa
            num_transactions = random.randint(3, min(MAX_SEQUENCE_LENGTH, settings.MAX_INDIVIDUAL_LENGTH - len(individual)))
            
            # Lấy 2-3 hàm từ các nguồn khác nhau
            sources = []
            
            # Thêm hàm từ optimal_sequence nếu có
            if self.optimal_sequences and len(self.optimal_sequences) > 0:
                seq = random.choice(self.optimal_sequences)
                if seq and len(seq) > 0:
                    func_name = random.choice(seq)
                    func_hash = self._get_function_hash_by_name(func_name)
                    if func_hash and func_hash in self.interface:
                        sources.append((func_hash, self.interface[func_hash]))
            
            # Thêm hàm từ critical_path nếu có
            if self.critical_paths and len(self.critical_paths) > 0:
                path = random.choice(self.critical_paths)
                if path and len(path) > 0:
                    func_name = random.choice(path)
                    func_hash = self._get_function_hash_by_name(func_name)
                    if func_hash and func_hash in self.interface:
                        sources.append((func_hash, self.interface[func_hash]))
            
            # Thêm 1-2 hàm ngẫu nhiên từ interface
            for _ in range(2):
                if available_functions:
                    random_func = random.choice(available_functions)
                    sources.append((random_func, self.interface[random_func]))
            
            # Shuffle các nguồn để đa dạng thứ tự
            random.shuffle(sources)
            
            # Chọn các hàm từ sources (không quá num_transactions)
            selected_sources = sources[:num_transactions]
            
            # Sinh transaction từ các nguồn đã chọn
            for func_hash, func_args_types in selected_sources:
                tx = super().generate_individual(func_hash, func_args_types, default_value=default_value)
                if tx:
                    individual.extend(tx)
                    # Đánh dấu là enhanced nếu từ optimal/critical, ngược lại là original
                    if any(self._get_function_hash_by_name(func_name) == func_hash 
                           for seq in self.optimal_sequences for func_name in seq) or \
                       any(self._get_function_hash_by_name(func_name) == func_hash 
                           for path in self.critical_paths for func_name in path):
                        rag_enhanced_functions += 1
                    else:
                        original_functions += 1
        
        # Nếu không có transaction nào được tạo hoặc chiến lược "random", sử dụng random nhưng với số lượng nhỏ
        if len(individual) <= (1 if self.generate_constructor() else 0) or strategy == "random":
            # Sinh ngẫu nhiên theo cách của generator gốc nhưng giới hạn số lượng
            num_transactions = random.randint(2, min(MAX_SEQUENCE_LENGTH, settings.MAX_INDIVIDUAL_LENGTH - len(individual)))
            
            # Dùng available_functions đã shuffle thay vì functions_pool để đảm bảo đa dạng
            functions_to_use = available_functions[:num_transactions] if len(available_functions) >= num_transactions else available_functions
            
            for function in functions_to_use:
                argument_types = self.interface[function]
                tx = super().generate_individual(function, argument_types, default_value=default_value)
                if tx:
                    individual.extend(tx)
                    original_functions += 1
        
        # Đảm bảo không vượt quá MAX_INDIVIDUAL_LENGTH
        if len(individual) > settings.MAX_INDIVIDUAL_LENGTH:
            # Giữ lại constructor và cắt bớt các transactions
            if self.generate_constructor():
                individual = individual[:1] + individual[1:settings.MAX_INDIVIDUAL_LENGTH]
            else:
                individual = individual[:settings.MAX_INDIVIDUAL_LENGTH]

        # Thêm chi tiết về hash của các hàm để debug
        func_names = []
        if len(individual) > 0:
            for idx, tx in enumerate(individual):
                if idx == 0 and "constructor" in tx.get("arguments", []):
                    func_names.append("constructor")
                    continue
                    
                if "arguments" in tx and len(tx["arguments"]) > 0:
                    func_hash = tx["arguments"][0]
                    func_name = "unknown"
                    if self.interface_mapper:
                        for name, hash in self.interface_mapper.items():
                            if hash == func_hash:
                                func_name = name.split("(")[0]  # Lấy tên hàm không kèm tham số
                                break
                    func_names.append(f"{func_name}({func_hash[:8]})")
        
        # Log thống kê để kiểm soát, bao gồm tên các hàm được gọi
        if len(func_names) > 0:
            self.logger.info(f"Generated sequence with {len(individual)} transactions ({rag_enhanced_functions} enhanced, {original_functions} original)")
            self.logger.info(f"Sequence details: {' -> '.join(func_names)}")
        else:
            self.logger.info(f"Generated empty sequence")
        
        # Lưu lại sequence tốt (chỉ khi có ít nhất một transaction ngoài constructor)
        if hasattr(self, 'good_sequences') and len(individual) > 1:
            if not isinstance(self.good_sequences, list):
                self.good_sequences = []
            if len(self.good_sequences) < 30:  # Giới hạn số lượng lưu trữ
                self.good_sequences.append(individual)
                    
        return individual

    def _fetch_rag_suggestion(self, prompt: str) -> Optional[str]:
        """
        Gọi API Flask để lấy gợi ý từ RAG
        
        :param prompt: Câu hỏi hoặc prompt gửi tới RAG
        :return: Kết quả từ RAG hoặc None nếu lỗi
        """
        max_retries = self.rag_max_retries
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                self.logger.info(f"Sending request to RAG server: {prompt[:50]}...")
                response = requests.post(
                    self.rag_api_endpoint,
                    json={"prompt": prompt},
                    timeout=self.rag_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.logger.info(f"RAG response received successfully")
                    
                    if "response" in result:
                        response_text = result["response"]
                        
                        # Kiểm tra nếu phản hồi là "Không có thông tin phù hợp"
                        if response_text == "Không có thông tin phù hợp":
                            self.logger.warning("RAG returned 'Không có thông tin phù hợp', using fallback values")
                            # Tạo JSON mặc định dựa trên loại prompt
                            if "Parameter type:" in prompt:
                                # Đây là prompt yêu cầu tham số cho hàm
                                param_type_match = re.search(r"Parameter type: ([a-zA-Z0-9\[\]]+)", prompt)
                                if param_type_match:
                                    param_type = param_type_match.group(1)
                                    if param_type.startswith("uint"):
                                        return "0"
                                    elif param_type.startswith("int"):
                                        return "0"
                                    elif param_type == "address":
                                        return "\"0x0000000000000000000000000000000000000000\""
                                    elif param_type == "bool":
                                        return "false"
                                    elif param_type.startswith("bytes"):
                                        return "\"0x00\""
                                    elif param_type == "string":
                                        return "\"\""
                                    else:
                                        return "null"
                            elif "Function:" in prompt:
                                # Đây là prompt yêu cầu tham số cho hàm cụ thể
                                function_match = re.search(r"Function: ([^\n]+)", prompt)
                                param_types_match = re.search(r"Parameter types: \[(.*?)\]", prompt)
                                
                                if function_match and param_types_match:
                                    function_name = function_match.group(1)
                                    param_types = param_types_match.group(1).split(", ")
                                    
                                    # Tạo mảng tham số mặc định dựa trên kiểu
                                    default_args = []
                                    for param_type in param_types:
                                        param_type = param_type.strip().strip("'\"")
                                        if param_type.startswith("uint"):
                                            default_args.append(0)
                                        elif param_type.startswith("int"):
                                            default_args.append(0)
                                        elif param_type == "address":
                                            default_args.append("0x0000000000000000000000000000000000000000")
                                        elif param_type == "bool":
                                            default_args.append(False)
                                        elif param_type.startswith("bytes"):
                                            default_args.append("0x00")
                                        elif param_type == "string":
                                            default_args.append("")
                                        else:
                                            default_args.append(None)
                                    
                                    return json.dumps(default_args)
                            
                            # Fallback cho các trường hợp khác
                            return "[]"
                        
                        # Kiểm tra nếu phản hồi có thể là JSON
                        try:
                            # Tìm chuỗi JSON trong phản hồi
                            json_match = re.search(r'\[.*?\]|\{.*?\}', response_text, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                # Thử parse JSON
                                json.loads(json_str)
                                return json_str
                            else:
                                # Nếu không tìm thấy chuỗi JSON, trả về nguyên phản hồi
                                return response_text
                        except json.JSONDecodeError:
                            # Nếu không thể parse JSON, trả về nguyên phản hồi
                            return response_text
                    else:
                        self.logger.warning(f"RAG response missing 'response' field: {result}")
                else:
                    self.logger.warning(f"RAG server returned status code {response.status_code}: {response.text[:100]}")
                
            except requests.ConnectionError:
                self.logger.warning(f"Connection error to RAG server (retry {retry_count+1}/{max_retries+1})")
            except requests.Timeout:
                self.logger.warning(f"Timeout connecting to RAG server (retry {retry_count+1}/{max_retries+1})")
            except Exception as e:
                self.logger.warning(f"Error fetching RAG suggestion: {str(e)} (retry {retry_count+1}/{max_retries+1})")
            
            retry_count += 1
            if retry_count <= max_retries:
                # Backoff tăng dần theo số lần thử
                wait_time = 2 * retry_count
                self.logger.info(f"Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
        
        # Không thể kết nối đến RAG server, thử lấy giá trị fallback
        self.logger.warning("All retries failed, using fallback value")
        return self._get_fallback_value(prompt)

    def _get_fallback_value(self, prompt: str) -> Optional[str]:
        """
        Phân tích prompt để trả về giá trị mặc định phù hợp với kiểu dữ liệu
        
        :param prompt: Prompt gốc được gửi cho RAG
        :return: Giá trị mặc định phù hợp với kiểu
        """
        try:
            # Trích xuất kiểu dữ liệu từ prompt
            type_match = re.search(r"Parameter type: ([a-zA-Z0-9\[\]]+)", prompt)
            if not type_match:
                return None
                
            param_type = type_match.group(1)
            
            # Trả về giá trị mặc định dựa trên kiểu
            if param_type.startswith("uint"):
                return "0"  # Giá trị uint an toàn
            elif param_type.startswith("int"):
                return "0"  # Giá trị int an toàn
            elif param_type == "address":
                return "0x0000000000000000000000000000000000000000"  # zero address
            elif param_type == "bool":
                return "false"
            elif param_type.startswith("bytes"):
                return "0x00"
            elif param_type == "string":
                return ""
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error in fallback generation: {e}")
            return None

    def initialize_population(self, size=10):
        """Khởi tạo quần thể các chuỗi giao dịch"""
        self.population = []
        for _ in range(size):
            individual = self.generate_random_individual()
            self.population.append({
                "individual": individual,
                "fitness": 0,  # Sẽ được cập nhật sau khi chạy
                "coverage": 0
            })
        self.logger.info(f"Initialized population with {size} individuals")

    def update_fitness(self, individual_index, coverage, found_bugs=0):
        """Cập nhật độ thích nghi của một cá thể trong quần thể sử dụng phương pháp từ CrossFuzz gốc"""
        # Tạo một file log riêng cho quá trình đánh giá fitness
        log_dir = os.path.join(os.getcwd(), "fuzzing_logs")
        os.makedirs(log_dir, exist_ok=True)
        fitness_log_file = os.path.join(log_dir, "fitness_evolution.txt")
        
        if hasattr(self, 'population') and individual_index < len(self.population):
            # Lấy thông tin môi trường từ settings
            old_fitness = None
            old_coverage = None
            
            # Lưu giá trị fitness và coverage cũ để so sánh
            if isinstance(self.population[individual_index], dict):
                old_fitness = self.population[individual_index].get("fitness", None)
                old_coverage = self.population[individual_index].get("coverage", None)
            else:
                old_fitness = getattr(self.population[individual_index], "fitness", None)
                old_coverage = getattr(self.population[individual_index], "coverage", None)
                
            if hasattr(settings, 'GLOBAL_ENV') and settings.GLOBAL_ENV:
                env = settings.GLOBAL_ENV
                indv = None
                
                # Lấy cá thể từ quần thể
                if isinstance(self.population[individual_index], dict) and "individual" in self.population[individual_index]:
                    indv = self.population[individual_index]["individual"]
                else:
                    indv = self.population[individual_index]
                
                # Xác định hash của individual (không phải mọi individual đều có hash)
                indv_hash = None
                if hasattr(indv, "hash"):
                    indv_hash = indv.hash
                
                # Tính fitness sử dụng phương pháp của CrossFuzz
                if indv_hash and hasattr(env, 'individual_branches') and indv_hash in env.individual_branches:
                    # Import các phương thức tính fitness
                    from fuzzer.engine.fitness import compute_branch_coverage_fitness, compute_data_dependency_fitness
                    
                    # Tính fitness theo cách của CrossFuzz (branch coverage)
                    fitness = compute_branch_coverage_fitness(env.individual_branches[indv_hash], env.code_coverage)
                    fitness_type = "branch_coverage_fitness"
                    
                    # Thêm data dependency nếu được bật
                    if hasattr(env.args, 'data_dependency') and env.args.data_dependency and hasattr(env, 'data_dependencies'):
                        data_dependency_fitness = compute_data_dependency_fitness(indv, env.data_dependencies)
                        fitness += data_dependency_fitness
                        fitness_type = "branch_coverage + data_dependency"
                    
                    # Lưu giá trị fitness và coverage
                    if isinstance(self.population[individual_index], dict):
                        self.population[individual_index]["fitness"] = fitness
                        self.population[individual_index]["coverage"] = coverage
                    else:
                        # Nếu cá thể không phải dict, tạo thuộc tính fitness và coverage
                        setattr(self.population[individual_index], "fitness", fitness)
                        setattr(self.population[individual_index], "coverage", coverage)
                    
                    # Ghi log sự thay đổi fitness
                    with open(fitness_log_file, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now()}] Individual {individual_index} fitness updated:\n")
                        f.write(f"  - Fitness type: {fitness_type}\n")
                        f.write(f"  - Old fitness: {old_fitness}\n")
                        f.write(f"  - New fitness: {fitness}\n")
                        f.write(f"  - Coverage: {coverage}\n")
                        f.write(f"  - Found bugs: {found_bugs}\n")
                        f.write(f"  - Individual hash: {indv_hash}\n")
                        # Ghi thông tin về sequence length nếu có
                        if hasattr(indv, "__len__"):
                            f.write(f"  - Sequence length: {len(indv)}\n")
                        f.write("\n")
                    
                    self.logger.info(f"Updated fitness of individual {individual_index} to {fitness:.4f} using CrossFuzz algorithm")
                    return
            
            # Nếu không thể dùng công thức gốc, sử dụng công thức đơn giản
            fitness = coverage * 0.7 + found_bugs * 0.3
            
            # Trong CrossFuzz gốc, fitness thấp hơn là tốt hơn (số lượng nhánh CHƯA được phủ)
            # Chuyển đổi để phù hợp: 1.0 / (1.0 + fitness)
            normalized_fitness = 1.0 / (1.0 + fitness)
            
            # Lưu giá trị fitness
            if isinstance(self.population[individual_index], dict):
                self.population[individual_index]["fitness"] = normalized_fitness
                self.population[individual_index]["coverage"] = coverage
            else:
                setattr(self.population[individual_index], "fitness", normalized_fitness)
                setattr(self.population[individual_index], "coverage", coverage)
            
            # Ghi log sự thay đổi fitness
            with open(fitness_log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] Individual {individual_index} fitness updated (alternative method):\n")
                f.write(f"  - Raw fitness components: coverage={coverage}, found_bugs={found_bugs}\n")
                f.write(f"  - Old fitness: {old_fitness}\n")
                f.write(f"  - New fitness: {normalized_fitness}\n")
                f.write(f"  - Coverage change: {old_coverage} -> {coverage}\n")
                f.write("\n")
            
            self.logger.warning(f"Using alternative fitness for individual {individual_index}: {fitness:.4f} -> {normalized_fitness:.4f}")
        else:
            self.logger.warning(f"Cannot update fitness - invalid individual index {individual_index} or no population")

    def crossover(self, parent1_index, parent2_index):
        """Lai ghép hai cá thể để tạo cá thể mới"""
        if not hasattr(self, 'population'):
            return self.generate_random_individual()
        
        if parent1_index >= len(self.population) or parent2_index >= len(self.population):
            return self.generate_random_individual()
        
        parent1 = self.population[parent1_index]["individual"]
        parent2 = self.population[parent2_index]["individual"]
        
        # Bỏ qua constructor vì nó nên giữ nguyên
        constructor = parent1[:1] if len(parent1) > 0 else []
        
        # Lấy phần còn lại của các cá thể cha mẹ
        parent1_txs = parent1[1:] if len(parent1) > 1 else []
        parent2_txs = parent2[1:] if len(parent2) > 1 else []
        
        if not parent1_txs or not parent2_txs:
            return self.generate_random_individual()
        
        # Chọn điểm cắt
        cut_point = random.randint(1, min(len(parent1_txs), len(parent2_txs)))
        
        # Tạo cá thể con bằng cách kết hợp các phần từ cha mẹ
        child_txs = parent1_txs[:cut_point] + parent2_txs[cut_point:]
        
        # Giới hạn số lượng giao dịch
        max_txs = settings.MAX_INDIVIDUAL_LENGTH - len(constructor)
        if len(child_txs) > max_txs:
            child_txs = child_txs[:max_txs]
        
        # Tạo cá thể hoàn chỉnh
        child = constructor + child_txs
        
        self.logger.info(f"Created new individual via crossover with {len(child)} transactions")
        return child

    def mutate(self, individual):
        """Đột biến một cá thể để tạo biến thể mới với độ đa dạng cao hơn"""
        # Tạo một file log riêng cho quá trình đột biến
        log_dir = os.path.join(os.getcwd(), "fuzzing_logs")
        os.makedirs(log_dir, exist_ok=True)
        mutation_log_file = os.path.join(log_dir, "mutation_statistics.txt")
        
        # Khởi tạo hoặc cập nhật biến theo dõi số lượng đột biến tổng cộng
        if not hasattr(self, '_mutation_stats_total'):
            self._mutation_stats_total = {
                "replace": 0,
                "insert": 0,
                "remove": 0,
                "swap": 0,
                "param_mutate": 0,
                "total_mutations": 0,
                "successful_mutations": 0
            }
        
        # Khởi tạo hoặc cập nhật số lượng cá thể đã đột biến
        if not hasattr(self, '_mutated_individuals_count'):
            self._mutated_individuals_count = 0
        self._mutated_individuals_count += 1
        
        # Giữ nguyên constructor
        constructor = individual[:1] if len(individual) > 0 else []
        transactions = individual[1:] if len(individual) > 1 else []
        
        if not transactions:
            # Ghi log khi không có transaction để đột biến
            with open(mutation_log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] MUTATION STOPPED: No transactions to mutate. Creating random individual instead.\n")
            self.logger.warning("Mutation stopped: No transactions to mutate. Creating random individual instead.")
            return self.generate_random_individual()
        
        # Ghi log bắt đầu quá trình đột biến
        original_tx_count = len(transactions)
        with open(mutation_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] MUTATION STARTED: Original sequence has {original_tx_count} transactions\n")
        
        # Lưu fitness trước khi đột biến nếu có
        original_fitness = None
        original_coverage = None
        if hasattr(self, 'population'):
            for item in self.population:
                if isinstance(item, dict) and "individual" in item and item["individual"] == individual:
                    original_fitness = item.get("fitness", None)
                    original_coverage = item.get("coverage", None)
                    break
        
        # Chọn ngẫu nhiên một số lượng giao dịch để đột biến
        # Tăng cường đột biến nhiều transaction hơn cho độ đa dạng cao
        max_possible_mutations = max(2, len(transactions) // 2)
        num_mutations = random.randint(1, max_possible_mutations)
        
        # Cập nhật số lượng đột biến dự kiến trong thống kê tổng thể
        self._mutation_stats_total["total_mutations"] += num_mutations
        
        # Ghi log số lượng đột biến sẽ thực hiện
        with open(mutation_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Planning to apply {num_mutations} mutations out of maximum {max_possible_mutations}\n")
        
        self.logger.info(f"Mutation process: Planning to apply {num_mutations} mutations on sequence of {len(transactions)} transactions")
        
        # Khởi tạo thống kê đột biến
        mutation_stats = {
            "replace": 0,
            "insert": 0,
            "remove": 0, 
            "swap": 0,
            "param_mutate": 0
        }
        
        # Theo dõi các đột biến đã thực hiện để debug
        mutations_applied = []
        
        # Đếm số lần đột biến đã thực hiện
        mutations_performed = 0
        
        for i in range(num_mutations):
            # Tùy chỉnh phân phối của các loại đột biến 
            # Ưu tiên "replace" và "swap" nhiều hơn để tăng độ đa dạng
            mutation_weights = [40, 20, 15, 25]  # Tỷ lệ % cho replace, insert, remove, swap
            mutation_type = random.choices(
                ["replace", "insert", "remove", "swap"],
                weights=mutation_weights
            )[0]
            
            mutation_success = False
            
            if mutation_type == "replace" and transactions:
                # Thay thế một giao dịch bằng giao dịch mới
                idx = random.randint(0, len(transactions) - 1)
                
                # Lấy danh sách hàm hiện có để tránh lặp lại
                existing_functions = set()
                for tx in transactions:
                    if "arguments" in tx and tx["arguments"]:
                        existing_functions.add(tx["arguments"][0])
                
                # Ưu tiên chọn một hàm mới chưa có trong sequence
                available_functions = [f for f in self.interface.keys() if f not in existing_functions]
                
                if available_functions and random.random() < 0.8:  # 80% thời gian chọn hàm mới
                    function = random.choice(available_functions)
                    argument_types = self.interface[function]
                else:
                    function, argument_types = self.get_random_function_with_argument_types()
                
                new_tx = self.generate_individual(function, argument_types)
                if new_tx:
                    old_func = transactions[idx].get("arguments", ["unknown"])[0]
                    transactions[idx] = new_tx[0]
                    mutations_applied.append(f"Replaced tx {idx+1}: {old_func[:8]} -> {function[:8]}")
                    mutation_stats["replace"] += 1
                    self._mutation_stats_total["replace"] += 1
                    mutation_success = True
            
            elif mutation_type == "insert" and len(transactions) < settings.MAX_INDIVIDUAL_LENGTH - 1:
                # Chèn một giao dịch mới
                # Ưu tiên chèn hàm mới chưa có trong sequence
                existing_functions = set()
                for tx in transactions:
                    if "arguments" in tx and tx["arguments"]:
                        existing_functions.add(tx["arguments"][0])
                
                available_functions = [f for f in self.interface.keys() if f not in existing_functions]
                
                if available_functions and random.random() < 0.8:  # 80% thời gian chọn hàm mới
                    function = random.choice(available_functions)
                    argument_types = self.interface[function]
                else:
                    function, argument_types = self.get_random_function_with_argument_types()
                
                new_tx = self.generate_individual(function, argument_types)
                if new_tx:
                    idx = random.randint(0, len(transactions))
                    transactions.insert(idx, new_tx[0])
                    mutations_applied.append(f"Inserted tx at position {idx+1}: {function[:8]}")
                    mutation_stats["insert"] += 1
                    self._mutation_stats_total["insert"] += 1
                    mutation_success = True
            
            elif mutation_type == "remove" and len(transactions) > 1:
                # Xóa một giao dịch
                idx = random.randint(0, len(transactions) - 1)
                func_to_remove = transactions[idx].get("arguments", ["unknown"])[0]
                transactions.pop(idx)
                mutations_applied.append(f"Removed tx {idx+1}: {func_to_remove[:8]}")
                mutation_stats["remove"] += 1
                self._mutation_stats_total["remove"] += 1
                mutation_success = True
            
            elif mutation_type == "swap" and len(transactions) > 1:
                # Đổi vị trí hai giao dịch
                idx1 = random.randint(0, len(transactions) - 1)
                idx2 = random.randint(0, len(transactions) - 1)
                while idx1 == idx2:  # Đảm bảo hai vị trí khác nhau
                    idx2 = random.randint(0, len(transactions) - 1)
                    
                func1 = transactions[idx1].get("arguments", ["unknown"])[0]
                func2 = transactions[idx2].get("arguments", ["unknown"])[0]
                
                transactions[idx1], transactions[idx2] = transactions[idx2], transactions[idx1]
                mutations_applied.append(f"Swapped tx {idx1+1}:{func1[:8]} <-> tx {idx2+1}:{func2[:8]}")
                mutation_stats["swap"] += 1
                self._mutation_stats_total["swap"] += 1
                mutation_success = True
            
            # Thêm thuật toán đột biến tham số - đột biến giá trị của transaction
            elif mutation_type == "param_mutate" and transactions and random.random() < 0.3:
                idx = random.randint(0, len(transactions) - 1)
                if "arguments" in transactions[idx]:
                    func_hash = transactions[idx]["arguments"][0]
                    func_args_types = self.interface.get(func_hash, [])
                    
                    # Sinh lại transaction với tham số mới
                    new_tx = self.generate_individual(func_hash, func_args_types)
                    if new_tx:
                        transactions[idx] = new_tx[0]
                        mutations_applied.append(f"Mutated params of tx {idx+1}: {func_hash[:8]}")
                        mutation_stats["param_mutate"] += 1
                        self._mutation_stats_total["param_mutate"] += 1
                        mutation_success = True
            
            if mutation_success:
                mutations_performed += 1
                self._mutation_stats_total["successful_mutations"] += 1
            else:
                # Ghi log khi đột biến thất bại
                with open(mutation_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] Mutation attempt {i+1} failed: Type={mutation_type}\n")
            
            # Kiểm tra điều kiện dừng: nếu không còn transaction sau khi xóa
            if len(transactions) == 0:
                with open(mutation_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] MUTATION STOPPED: All transactions were removed\n")
                self.logger.warning("Mutation stopped: All transactions were removed")
                # Thêm lại ít nhất một transaction mới
                function, argument_types = self.get_random_function_with_argument_types()
                new_tx = self.generate_individual(function, argument_types)
                if new_tx:
                    transactions.extend(new_tx)
                break
        
        # Tạo cá thể mới sau đột biến
        mutated_individual = constructor + transactions
        
        # So sánh số lượng transaction sau khi đột biến
        final_tx_count = len(transactions)
        tx_change = final_tx_count - original_tx_count
        
        # Ghi log tổng kết đột biến
        with open(mutation_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] MUTATION COMPLETED:\n")
            f.write(f"  - Planned mutations: {num_mutations}\n")
            f.write(f"  - Mutations performed: {mutations_performed}\n")
            f.write(f"  - Transaction count: {original_tx_count} -> {final_tx_count} ({tx_change:+d})\n")
            f.write(f"  - Mutation statistics: {json.dumps(mutation_stats)}\n")
            f.write(f"  - Applied mutations: {'; '.join(mutations_applied)}\n")
            if original_fitness is not None:
                f.write(f"  - Original fitness: {original_fitness}\n")
            if original_coverage is not None:
                f.write(f"  - Original coverage: {original_coverage}\n")
            f.write("\n")
        
        # Log thông tin chi tiết về quá trình đột biến
        if mutations_applied:
            self.logger.info(f"Applied {len(mutations_applied)} mutations: {'; '.join(mutations_applied)}")
            self.logger.info(f"Mutation statistics: Replace={mutation_stats['replace']}, Insert={mutation_stats['insert']}, "
                            f"Remove={mutation_stats['remove']}, Swap={mutation_stats['swap']}, "
                            f"ParamMutate={mutation_stats['param_mutate']}")
            self.logger.info(f"Transaction count changed from {original_tx_count} to {final_tx_count} ({tx_change:+d})")
        else:
            self.logger.warning("No mutations were applied")
        
        return mutated_individual

    def select_parents(self):
        """Chọn cha mẹ để lai ghép dựa trên độ thích nghi"""
        if not hasattr(self, 'population') or len(self.population) < 2:
            return 0, 0
        
        # Tính tổng độ thích nghi
        total_fitness = sum(item["fitness"] for item in self.population)
        
        # Nếu tổng độ thích nghi là 0, chọn ngẫu nhiên
        if total_fitness == 0:
            return random.randint(0, len(self.population) - 1), random.randint(0, len(self.population) - 1)
        
        # Chọn cha mẹ dựa trên độ thích nghi (roulette wheel selection)
        probabilities = [item["fitness"] / total_fitness for item in self.population]
        parent1_idx = random.choices(range(len(self.population)), weights=probabilities)[0]
        
        # Đảm bảo parent2 khác parent1
        remaining_indices = list(range(len(self.population)))
        remaining_indices.remove(parent1_idx)
        remaining_probs = [probabilities[i] for i in remaining_indices]
        
        # Chuẩn hóa lại xác suất
        total_remaining = sum(remaining_probs)
        if total_remaining > 0:
            remaining_probs = [p / total_remaining for p in remaining_probs]
        else:
            remaining_probs = [1.0 / len(remaining_indices)] * len(remaining_indices)
        
        parent2_idx = random.choices(remaining_indices, weights=remaining_probs)[0]
        
        return parent1_idx, parent2_idx

    def evolve_population(self):
        """Phát triển quần thể qua một thế hệ"""
        # Tạo một file log riêng cho quá trình tiến hóa
        log_dir = os.path.join(os.getcwd(), "fuzzing_logs")
        os.makedirs(log_dir, exist_ok=True)
        evolution_log_file = os.path.join(log_dir, "evolution_process.txt")
        
        if not hasattr(self, 'population') or len(self.population) < 2:
            with open(evolution_log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] EVOLUTION: Population not initialized or too small. Initializing new population.\n")
            self.initialize_population()
            return self.generate_random_individual()
        
        # Sắp xếp quần thể theo độ thích nghi
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Lưu thông tin về fitness của quần thể hiện tại
        population_stats = {
            "best_fitness": self.population[0]["fitness"] if isinstance(self.population[0], dict) else getattr(self.population[0], "fitness", 0),
            "worst_fitness": self.population[-1]["fitness"] if isinstance(self.population[-1], dict) else getattr(self.population[-1], "fitness", 0),
            "avg_fitness": sum(x["fitness"] if isinstance(x, dict) else getattr(x, "fitness", 0) for x in self.population) / len(self.population),
            "population_size": len(self.population)
        }
        
        # Chọn chiến lược: lai ghép, đột biến hoặc giữ nguyên cá thể tốt nhất
        strategy = random.choices(
            ["elite", "crossover", "mutate", "random"], 
            weights=[0.1, 0.5, 0.3, 0.1]
        )[0]
        
        # Ghi log bắt đầu quá trình tiến hóa
        with open(evolution_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] EVOLUTION STARTED:\n")
            f.write(f"  - Strategy: {strategy}\n")
            f.write(f"  - Population size: {len(self.population)}\n")
            f.write(f"  - Best fitness: {population_stats['best_fitness']}\n")
            f.write(f"  - Worst fitness: {population_stats['worst_fitness']}\n")
            f.write(f"  - Average fitness: {population_stats['avg_fitness']}\n")
        
        # Đếm số lần sử dụng mỗi chiến lược
        if not hasattr(self, '_evolution_strategy_counts'):
            self._evolution_strategy_counts = {
                "elite": 0, 
                "crossover": 0, 
                "mutate": 0, 
                "random": 0
            }
        self._evolution_strategy_counts[strategy] += 1
        
        result_individual = None
        
        if strategy == "elite":
            # Giữ nguyên cá thể tốt nhất
            self.logger.info("Using elite individual")
            result_individual = self.population[0]["individual"]
            with open(evolution_log_file, "a", encoding="utf-8") as f:
                f.write(f"  - Elite strategy: Selected individual with fitness {self.population[0]['fitness']}\n")
        
        elif strategy == "crossover":
            # Lai ghép hai cá thể
            parent1_idx, parent2_idx = self.select_parents()
            with open(evolution_log_file, "a", encoding="utf-8") as f:
                parent1_fitness = self.population[parent1_idx]["fitness"] if isinstance(self.population[parent1_idx], dict) else getattr(self.population[parent1_idx], "fitness", 0)
                parent2_fitness = self.population[parent2_idx]["fitness"] if isinstance(self.population[parent2_idx], dict) else getattr(self.population[parent2_idx], "fitness", 0)
                f.write(f"  - Crossover strategy: Selected parents {parent1_idx} (fitness: {parent1_fitness}) and {parent2_idx} (fitness: {parent2_fitness})\n")
            
            result_individual = self.crossover(parent1_idx, parent2_idx)
        
        elif strategy == "mutate":
            # Đột biến một cá thể tốt
            elite_idx = random.randint(0, min(3, len(self.population) - 1))
            with open(evolution_log_file, "a", encoding="utf-8") as f:
                elite_fitness = self.population[elite_idx]["fitness"] if isinstance(self.population[elite_idx], dict) else getattr(self.population[elite_idx], "fitness", 0)
                f.write(f"  - Mutation strategy: Selected elite individual {elite_idx} (fitness: {elite_fitness}) for mutation\n")
            
            result_individual = self.mutate(self.population[elite_idx]["individual"])
        
        else:
            # Tạo cá thể hoàn toàn mới
            with open(evolution_log_file, "a", encoding="utf-8") as f:
                f.write(f"  - Random strategy: Creating completely new individual\n")
            
            result_individual = self.generate_random_individual()
        
        # Ghi log kết thúc quá trình tiến hóa
        with open(evolution_log_file, "a", encoding="utf-8") as f:
            f.write(f"  - Result: Individual created with {len(result_individual) if result_individual else 0} transactions\n")
            f.write(f"  - Evolution statistics: {json.dumps(self._evolution_strategy_counts)}\n")
            f.write("\n")
        
        return result_individual

    def log_generation_summary(self, generation_number, coverage, branch_coverage, transactions_count):
        """
        Ghi log tóm tắt thông tin của generation hiện tại
        """
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = time.time()
            self._last_transactions_count = 0
        
        # Tính tốc độ transaction/giây
        current_time = time.time()
        time_diff = current_time - self._last_log_time
        trans_diff = transactions_count - self._last_transactions_count
        
        tps = trans_diff / time_diff if time_diff > 0 else 0
        
        # Tạo dòng phân cách
        separator = "=" * 40
        
        # Tạo bảng thống kê nhỏ gọn
        self.logger.info(f"\n{separator}")
        self.logger.info(f"GENERATION {generation_number} SUMMARY")
        self.logger.info(f"Code Coverage:    {coverage:.2f}%")
        self.logger.info(f"Branch Coverage:  {branch_coverage:.2f}%")
        self.logger.info(f"Transactions:     {transactions_count} (+{trans_diff})")
        self.logger.info(f"Speed:            {tps:.2f} tx/s")
        
        # Thêm thống kê về đột biến
        try:
            mutation_stats = self.log_mutation_stats(generation_number)
            if mutation_stats:
                success_rate = (mutation_stats["successful_mutations"] / mutation_stats["total_mutations"] * 100) if mutation_stats["total_mutations"] > 0 else 0
                self.logger.info(f"Mutations:        {mutation_stats['successful_mutations']}/{mutation_stats['total_mutations']} ({success_rate:.2f}%)")
        except Exception as e:
            self.logger.warning(f"Error logging mutation stats: {e}")
        
        self.logger.info(f"{separator}")
        
        # Cập nhật giá trị lần log trước
        self._last_log_time = current_time
        self._last_transactions_count = transactions_count
        
        # Thực hiện trace các chiến lược đang được sử dụng
        strategy_counts = getattr(self, '_strategy_counts', {
            "optimal_sequence": 0, 
            "critical_path": 0, 
            "mutation": 0, 
            "random": 0, 
            "combined": 0
        })
        
        self.logger.info("Strategy usage:")
        for strategy, count in strategy_counts.items():
            self.logger.info(f"  - {strategy}: {count}")
        
        # Ghi thông tin về transaction lengths
        if hasattr(self, 'population') and len(self.population) > 0:
            trans_lengths = []
            for i in range(min(5, len(self.population))):
                if hasattr(self.population[i], "individual"):
                    seq = self.population[i]["individual"] 
                elif hasattr(self.population[i], "chromosome"):
                    seq = self.population[i].chromosome
                else:
                    seq = self.population[i]
                
                if seq:
                    trans_lengths.append(len(seq))
            
            if trans_lengths:
                avg_length = sum(trans_lengths) / len(trans_lengths)
                self.logger.info(f"Avg sequence length (top 5): {avg_length:.2f}")
        
        return

    def save_detector_results(self, detector_results, output_file="detector_results.json"):
        """
        Lưu kết quả từ detector vào file JSON
        
        :param detector_results: Kết quả từ detector
        :param output_file: Đường dẫn file để lưu kết quả
        """
        try:
            # Định dạng dữ liệu cho file JSON
            results_to_save = []
            
            # Sắp xếp các lỗi theo loại
            vulnerabilities_by_type = {}
            
            for vuln_type, vulnerabilities in detector_results.items():
                if not isinstance(vulnerabilities, list):
                    continue
                    
                for vuln in vulnerabilities:
                    # Lấy thông tin về swc_id và severity nếu có
                    swc_id = vuln.get("swc_id", "Unknown")
                    severity = vuln.get("severity", "Unknown")
                    
                    # Lấy thông tin về sequence gây ra lỗi
                    transaction_sequence = []
                    if "transaction_sequence" in vuln:
                        for tx in vuln["transaction_sequence"]:
                            tx_info = {
                                "function": tx.get("function", "Unknown"),
                                "from": tx.get("from", "Unknown"),
                                "to": tx.get("to", "Unknown"),
                                "value": tx.get("value", 0),
                                "arguments": tx.get("arguments", [])
                            }
                            transaction_sequence.append(tx_info)
                    
                    # Chuẩn bị entry để lưu
                    entry = {
                        "vulnerability_type": vuln_type,
                        "swc_id": swc_id,
                        "severity": severity,
                        "description": vuln.get("description", ""),
                        "transaction_sequence": transaction_sequence,
                        "code_location": vuln.get("code_location", {})
                    }
                    
                    # Thêm vào danh sách theo loại
                    if vuln_type not in vulnerabilities_by_type:
                        vulnerabilities_by_type[vuln_type] = []
                    vulnerabilities_by_type[vuln_type].append(entry)
            
            # Thêm tổng kết
            summary = {
                "total_vulnerabilities": sum(len(vulns) for vulns in vulnerabilities_by_type.values()),
                "vulnerabilities_by_type": {k: len(v) for k, v in vulnerabilities_by_type.items()},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Cấu trúc cuối cùng để lưu
            final_result = {
                "summary": summary,
                "vulnerabilities": vulnerabilities_by_type
            }
            
            # Ghi vào file
            with open(output_file, 'w') as f:
                json.dump(final_result, f, indent=2, default=str)
            
            self.logger.info(f"Detector results saved to {output_file}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving detector results: {e}")
            return False

    def finalize_fuzzing(self, results, output_dir="./results"):
        """
        Thực hiện các tác vụ cuối cùng khi hoàn thành quá trình fuzzing
        
        :param results: Kết quả từ fuzzing engine
        :param output_dir: Thư mục lưu kết quả
        """
        import os
        
        # Tạo thư mục kết quả nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu kết quả detector 
        detector_file = os.path.join(output_dir, "detector_results.json")
        if "errors" in results:
            self.save_detector_results(results["errors"], detector_file)
        
        # Lưu thống kê tổng quan
        stats_file = os.path.join(output_dir, "fuzzing_stats.json")
        try:
            statistics = {
                "contract_name": self.contract_name,
                "fuzzer_type": "RAGEnhanced",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "coverage": {
                    "code_coverage": results.get("code_coverage", 0),
                    "branch_coverage": results.get("branch_coverage", 0)
                },
                "transactions": {
                    "total": results.get("total_transactions", 0),
                    "unique": results.get("unique_transactions", 0)
                },
                "execution_time": results.get("execution_time", 0),
                "memory_usage": results.get("memory_usage", 0),
                "strategy_usage": getattr(self, '_strategy_counts', {})
            }
            
            # Ghi vào file
            with open(stats_file, 'w') as f:
                json.dump(statistics, f, indent=2, default=str)
                
            self.logger.info(f"Fuzzing statistics saved to {stats_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving fuzzing statistics: {e}")
        
        # In tóm tắt cuối cùng
        self._print_final_summary(results)
        
        return True
    
    def _print_final_summary(self, results):
        """
        In tóm tắt kết quả cuối cùng
        
        :param results: Kết quả fuzzing
        """
        # Tạo đường viền
        border = "=" * 50
        
        # In header
        self.logger.info(f"\n{border}")
        self.logger.info(f"RAG ENHANCED FUZZING SUMMARY FOR {self.contract_name}")
        self.logger.info(f"{border}")
        
        # In thông tin độ phủ
        code_coverage = results.get("code_coverage", 0)
        branch_coverage = results.get("branch_coverage", 0)
        self.logger.info(f"CODE COVERAGE:     {code_coverage:.2f}%")
        self.logger.info(f"BRANCH COVERAGE:   {branch_coverage:.2f}%")
        
        # In thông tin giao dịch
        total_txs = results.get("total_transactions", 0)
        unique_txs = results.get("unique_transactions", 0)
        self.logger.info(f"TOTAL TRANSACTIONS:   {total_txs}")
        self.logger.info(f"UNIQUE TRANSACTIONS:  {unique_txs}")
        self.logger.info(f"TRANSACTION DIVERSITY: {(unique_txs/total_txs)*100:.2f}% unique")
        
        # In thông tin sử dụng chiến lược
        if hasattr(self, '_strategy_counts'):
            self.logger.info(f"{border}")
            self.logger.info("STRATEGY USAGE:")
            total_usage = sum(self._strategy_counts.values())
            for strategy, count in self._strategy_counts.items():
                percentage = (count / total_usage) * 100 if total_usage > 0 else 0
                self.logger.info(f"  - {strategy.upper()}: {count} ({percentage:.1f}%)")
        
        # In thông tin về lỗi phát hiện được
        if "errors" in results:
            error_count = sum(1 for error_list in results["errors"].values() 
                             if isinstance(error_list, list) for _ in error_list)
            self.logger.info(f"{border}")
            self.logger.info(f"DETECTED VULNERABILITIES: {error_count}")
            
            for error_type, errors in results["errors"].items():
                if isinstance(errors, list) and errors:
                    self.logger.info(f"  - {error_type}: {len(errors)}")
        
        # In footer
        self.logger.info(f"{border}")
        self.logger.info(f"FUZZING COMPLETED SUCCESSFULLY")
        self.logger.info(f"{border}\n")

    def generate_constructor(self):
        """
        Tạo constructor cho contract hiện tại trong sequence
        """
        if not self.interface or 'constructor' not in self.interface:
            return []

        # Sử dụng cách triển khai từ lớp gốc 
        return super().generate_constructor()
    
    def create_fake_accounts(self, instrumented_evm):
        """
        Tạo các tài khoản giả cho quá trình testing
        """
        # Thêm một số địa chỉ thường được sử dụng
        accounts = [
            "0xcafebabecafebabecafebabecafebabecafebabe",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "0x3333333333333333333333333333333333333333",
            "0x4444444444444444444444444444444444444444"
        ]
        
        # Tạo các tài khoản giả
        for address in accounts:
            instrumented_evm.create_fake_account(address)
            
        self.logger.info(f"Created {len(accounts)} fake accounts for testing")
        
        return accounts

    def log_mutation_stats(self, generation_number):
        """
        Ghi log thống kê về các đột biến đã thực hiện trong mỗi thế hệ
        
        :param generation_number: Số thứ tự của thế hệ hiện tại
        """
        # Tạo đường dẫn để lưu log
        log_dir = os.path.join(os.getcwd(), "fuzzing_logs")
        os.makedirs(log_dir, exist_ok=True)
        mutation_stats_file = os.path.join(log_dir, "mutation_stats_by_generation.txt")
        
        # Thu thập dữ liệu thống kê nếu có
        mutation_stats = getattr(self, '_mutation_stats_total', {
            "replace": 0,
            "insert": 0,
            "remove": 0,
            "swap": 0,
            "param_mutate": 0,
            "total_mutations": 0,
            "successful_mutations": 0
        })
        
        # Đếm số cá thể đã đột biến
        mutated_individuals = getattr(self, '_mutated_individuals_count', 0)
        
        # Ghi thống kê vào file
        with open(mutation_stats_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] GENERATION {generation_number} MUTATION STATISTICS:\n")
            f.write(f"  - Total individuals mutated: {mutated_individuals}\n")
            f.write(f"  - Total mutations attempted: {mutation_stats['total_mutations']}\n")
            f.write(f"  - Successful mutations: {mutation_stats['successful_mutations']} ({(mutation_stats['successful_mutations']/mutation_stats['total_mutations']*100) if mutation_stats['total_mutations'] > 0 else 0:.2f}%)\n")
            f.write(f"  - Replace operations: {mutation_stats['replace']}\n")
            f.write(f"  - Insert operations: {mutation_stats['insert']}\n")
            f.write(f"  - Remove operations: {mutation_stats['remove']}\n")
            f.write(f"  - Swap operations: {mutation_stats['swap']}\n")
            f.write(f"  - Parameter mutations: {mutation_stats['param_mutate']}\n")
            f.write("\n")
        
        self.logger.info(f"Generation {generation_number} mutation stats: {mutation_stats['successful_mutations']}/{mutation_stats['total_mutations']} successful mutations")
        return mutation_stats

def create_rag_enhanced_generator(
        interface: Dict, 
        bytecode: str,
        accounts: List[str],
        contract: str,
        api_key: str,
        analysis_result: Optional[Dict] = None,
        contract_name: Optional[str] = None,
        sol_path: Optional[str] = None,
        other_generators=None,
        interface_mapper=None) -> RAGEnhancedGenerator:
    """
    Hàm tiện ích để tạo RAGEnhanced"""
    return RAGEnhancedGenerator(
        interface=interface,
        bytecode=bytecode,
        accounts=accounts,
        contract=contract,
        api_key=api_key,
        analysis_result=analysis_result,
        contract_name=contract_name,
        sol_path=sol_path,
        other_generators=other_generators,
        interface_mapper=interface_mapper
    )
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
        super().__init__(interface, bytecode, accounts, contract, 
                        other_generators=other_generators, 
                        interface_mapper=interface_mapper,
                        contract_name=contract_name, 
                        sol_path=sol_path)
        
        self.api_key = api_key
        self.logger = initialize_logger("RAGEnhancedGenerator")
        
        # Lưu kết quả phân tích dataflow
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
        
        # Theo dõi transaction sequences đã được tạo
        self.generated_sequences = []
        
        # Cache cho các giá trị tham số
        self.arg_cache = {}
        
        # Cấu hình kết nối đến Flask RAG Server
        self.rag_api_endpoint = "http://localhost:5000/request"
        self.rag_timeout = 60  # seconds
        self.rag_max_retries = 3  # Số lần thử lại tối đa
        
        # Thống kê hiệu suất RAG
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
            
            # Xây dựng prompt cho RAG
            prompt = f"""
Analyze my smart contract and help me generate test values that might trigger vulnerabilities.

Function: {function_name}
Parameter types: {argument_types}

I need VALUES ONLY that would be good for fuzzing this function, focusing on potential security issues.
For each parameter type, return a suitable value that could expose vulnerabilities:

1. For uint/int: return numbers that might cause overflows or underflows 
2. For address: return specific addresses that might cause issues
3. For bool: return true or false based on which is more likely to cause issues
4. For bytes/string: return values that might cause issues

Return ONLY a JSON array containing appropriate values, one for each parameter.
Example: ["0xabc123...", 1000000]

Do not include explanations, types, or other text.
            """
            
            # Thêm thông tin về lỗ hổng nếu có
            if related_vulnerabilities:
                vuln_info = json.dumps(related_vulnerabilities, indent=2)
                prompt += f"""
Potential vulnerabilities:
{vuln_info}
                """
            
            # Thêm thông tin dataflow nếu có
            if dataflow_context:
                flow_info = json.dumps(dataflow_context, indent=2)
                prompt += f"""
Dataflow context:
{flow_info}
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
Function code:
{function_matches[0]}
                        """
                except Exception as e:
                    self.logger.warning(f"Could not read sol file: {e}")
            
            self.logger.info(f"Requesting argument values from RAG for {function_name}")
            
            # Gọi RAG API
            rag_response = self._fetch_rag_suggestion(prompt)
            
            if not rag_response:
                self.rag_failures += 1
                self.logger.warning(f"RAG returned no response for {function_name}")
                return None
                
            # Xử lý phản hồi
            try:
                # Làm sạch chuỗi JSON - xóa các format code blocks nếu có
                clean_response = re.sub(r"```json\s*|\s*```", "", rag_response).strip()
                
                # Xử lý trường hợp phản hồi có text bên ngoài JSON array
                json_array_match = re.search(r"\[\s*.*?\s*\]", clean_response, re.DOTALL)
                if json_array_match:
                    clean_response = json_array_match.group(0)
                
                args = json.loads(clean_response)
                
                # Đảm bảo args là list
                if not isinstance(args, list):
                    raise ValueError(f"Expected list response, got {type(args)}")
                
                # Xử lý đặc biệt cho các giá trị lồng nhau
                processed_args = []
                for i, arg in enumerate(args):
                    # Nếu arg là list (nested list), lấy phần tử đầu tiên
                    if isinstance(arg, list):
                        if len(arg) > 0:
                            processed_args.append(arg[0])
                        else:
                            # Fallback nếu list rỗng
                            if i < len(argument_types):
                                processed_args.append(self._get_default_value_for_type(argument_types[i]))
                            else:
                                processed_args.append(None)
                    else:
                        processed_args.append(arg)
                
                self.logger.info(f"RAG suggested args for {function_name}: {processed_args}")
                
                # Lưu vào cache
                self.arg_cache[cache_key] = processed_args
                self.rag_successes += 1
                
                return processed_args
                
            except json.JSONDecodeError as e:
                self.rag_failures += 1
                self.logger.error(f"JSON parse error for {function_name}: {e}")
                self.logger.error(f"RAG response: {rag_response}")
                return None
            except Exception as e:
                self.rag_failures += 1
                self.logger.error(f"Error processing RAG response for {function_name}: {e}")
                return None
            
        except Exception as e:
            self.rag_failures += 1
            self.logger.error(f"Error getting function args from RAG: {e}")
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
    
    def get_random_argument(self, type_str: str, function: str, argument_index: int) -> Any:
        """Override để sinh tham số tối ưu từ RAG khi có thể"""
        # Tìm tên hàm từ hash
        function_name = None
        for fname, fhash in self.interface_mapper.items() if self.interface_mapper else {}:
            if fhash == function:
                function_name = fname.split("(")[0]  # Lấy tên không có tham số
                break
        
        if function_name:
            # Thử lấy tham số từ RAG
            rag_args = self._get_function_args_from_rag(
                function_name, 
                function, 
                [type_str]
            )
            
            if rag_args and len(rag_args) > argument_index:
                arg_value = rag_args[argument_index]
                self.logger.info(f"Using RAG value for {function_name}.arg{argument_index}: {arg_value}")
                
                # Đảm bảo giá trị được chuyển đổi sang định dạng hợp lệ
                if type_str.startswith(("uint", "int")) and isinstance(arg_value, str):
                    # Chuyển đổi chuỗi số hoặc biểu thức thành số nguyên
                    try:
                        if arg_value.startswith("0x"):
                            return int(arg_value, 16)
                        else:
                            # Xử lý các biểu thức như 2**256-1
                            if "**" in arg_value or "-" in arg_value or "+" in arg_value or "*" in arg_value:
                                # Cẩn thận với eval() - chỉ dùng cho biểu thức số học đơn giản
                                cleaned_expr = re.sub(r"[^0-9\s\+\-\*\/\(\)\^]", "", arg_value.replace("**", "^"))
                                cleaned_expr = cleaned_expr.replace("^", "**")
                                if cleaned_expr:
                                    return eval(cleaned_expr)
                            return int(arg_value)
                    except Exception as e:
                        self.logger.warning(f"Error converting RAG value '{arg_value}' to int: {e}")
                
                return arg_value
        
        # Fallback sang phương thức mặc định
        return super().get_random_argument(type_str, function, argument_index)
    
    def generate_individual(self, function: str, argument_types: List[str], 
                           vuln_type: Optional[str] = None, default_value: bool = False) -> List[Dict]:
        """Override để sinh tham số tối ưu từ RAG"""
        # Tìm tên hàm từ hash
        function_name = None
        for fname, fhash in self.interface_mapper.items() if self.interface_mapper else {}:
            if fhash == function:
                function_name = fname.split("(")[0]  # Lấy tên không có tham số
                break
        
        # Nếu có tên hàm, thử lấy tham số từ RAG
        if function_name:
            rag_args = self._get_function_args_from_rag(
                function_name, 
                function, 
                argument_types,
                vuln_type
            )
            
            if rag_args:
                individual = []
                
                arguments = [function]  # Function selector là tham số đầu tiên
                for index, arg_type in enumerate(argument_types):
                    # Dùng giá trị từ RAG nếu có
                    if index < len(rag_args):
                        # Chuyển đổi giá trị RAG sang định dạng phù hợp
                        arg_value = rag_args[index]
                        
                        # Kiểm tra xem arg_value có phải là list không
                        if isinstance(arg_value, list):
                            self.logger.warning(f"Nested list detected for {function_name}, arg {index}: {arg_value}")
                            # Lấy phần tử đầu tiên nếu arg_value là list
                            arg_value = arg_value[0] if arg_value else None
                        
                        if arg_value is None:
                            arguments.append(self.get_random_argument(arg_type, function, index))
                        elif arg_type == "address" and isinstance(arg_value, str):
                            # Xử lý đặc biệt cho địa chỉ để tránh lỗi AddressEncoder
                            try:
                                if arg_value.startswith("0x"):
                                    # Kiểm tra độ dài địa chỉ
                                    if len(arg_value) == 42:  # Địa chỉ Ethereum đầy đủ (0x + 40 ký tự hex)
                                        # Chuyển địa chỉ thành dạng chuẩn để tránh lỗi AddressEncoder
                                        from eth_utils import to_checksum_address
                                        try:
                                            checksum_address = to_checksum_address(arg_value)
                                            arguments.append(checksum_address)
                                        except Exception:
                                            # Nếu không thể chuyển đổi, sử dụng một địa chỉ từ accounts pool
                                            if len(self.accounts) > 0:
                                                arguments.append(random.choice(self.accounts))
                                            else:
                                                arguments.append(self.get_random_argument(arg_type, function, index))
                                    else:
                                        # Địa chỉ không đúng định dạng, sử dụng một địa chỉ từ accounts pool
                                        if len(self.accounts) > 0:
                                            arguments.append(random.choice(self.accounts))
                                        else:
                                            arguments.append(self.get_random_argument(arg_type, function, index))
                                else:
                                    # Không phải địa chỉ hex, sử dụng một địa chỉ từ accounts pool
                                    if len(self.accounts) > 0:
                                        arguments.append(random.choice(self.accounts))
                                    else:
                                        arguments.append(self.get_random_argument(arg_type, function, index))
                            except Exception as e:
                                self.logger.warning(f"Error processing address: {e}")
                                # Fallback an toàn
                                if len(self.accounts) > 0:
                                    arguments.append(random.choice(self.accounts))
                                else:
                                    arguments.append(self.get_random_argument(arg_type, function, index))
                        elif arg_type.startswith(("uint", "int")) and isinstance(arg_value, str):
                            try:
                                if arg_value.startswith("0x"):
                                    arguments.append(int(arg_value, 16))
                                else:
                                    # Xử lý các biểu thức như 2**256-1
                                    if "**" in arg_value or "-" in arg_value or "+" in arg_value or "*" in arg_value:
                                        cleaned_expr = re.sub(r"[^0-9\s\+\-\*\/\(\)\^]", "", arg_value.replace("**", "^"))
                                        cleaned_expr = cleaned_expr.replace("^", "**")
                                        if cleaned_expr:
                                            arguments.append(eval(cleaned_expr))
                                        else:
                                            arguments.append(int(arg_value) if arg_value.isdigit() else self.get_random_argument(arg_type, function, index))
                                    else:
                                        arguments.append(int(arg_value) if arg_value.isdigit() else arg_value)
                            except Exception as e:
                                self.logger.warning(f"Error converting RAG value '{arg_value}' to int: {e}")
                                arguments.append(self.get_random_argument(arg_type, function, index))
                        else:
                            arguments.append(arg_value)
                    else:
                        arguments.append(self.get_random_argument(arg_type, function, index))
                
                try:
                    # Tạo giao dịch và thêm các thông tin khác
                    individual.append({
                        "account": self.get_random_account(function),
                        "contract": self.contract,
                        "amount": self.get_random_amount(function),
                        "arguments": arguments,
                        "blocknumber": self.get_random_blocknumber(function),
                        "timestamp": self.get_random_timestamp(function),
                        "gaslimit": self.get_random_gaslimit(function),
                        "call_return": dict(),
                        "extcodesize": dict(),
                        "returndatasize": dict()
                    })
                    
                    # Thêm các thông tin khác
                    address, call_return_value = self.get_random_callresult_and_address(function)
                    individual[-1]["call_return"] = {address: call_return_value}
                    
                    address, extcodesize_value = self.get_random_extcodesize_and_address(function)
                    individual[-1]["extcodesize"] = {address: extcodesize_value}
                    
                    address, value = self.get_random_returndatasize_and_address(function)
                    individual[-1]["returndatasize"] = {address: value}
                    
                    return individual
                except Exception as e:
                    self.logger.error(f"Error creating transaction: {e}")
        
        # Nếu không có thông tin từ RAG, dùng phương thức mặc định
        return super().generate_individual(function, argument_types, default_value)
    
    def generate_random_individual(self, func_hash=None, func_args_types=None, default_value=False):
        """Sinh chuỗi transaction tối ưu dựa trên phân tích và RAG"""
        # Nếu đã chỉ định hash và args, dùng chúng
        if func_hash is not None and func_args_types is not None:
            individual = []
            individual.extend(self.generate_constructor())
            individual.extend(self.generate_individual(func_hash, func_args_types, default_value=default_value))
            return individual
        
        # Chọn ngẫu nhiên một trong các chiến lược sinh sequence:
        strategy_weights = {
            "optimal": 0.4,      # 40% xác suất dùng sequence tối ưu 
            "vulnerability": 0.3, # 30% xác suất nhắm vào lỗ hổng
            "critical_path": 0.2, # 20% xác suất theo critical path
            "random": 0.1        # 10% xác suất sinh ngẫu nhiên
        }
        
        strategies = list(strategy_weights.keys())
        weights = list(strategy_weights.values())
        
        # Điều chỉnh trọng số dựa trên dữ liệu có sẵn
        if not self.optimal_sequences:
            weights[strategies.index("optimal")] = 0.1
            weights[strategies.index("random")] = 0.4
        if not self.potential_vulnerabilities:
            weights[strategies.index("vulnerability")] = 0.1
            weights[strategies.index("random")] = weights[strategies.index("random")] + 0.2
        if not self.critical_paths:
            weights[strategies.index("critical_path")] = 0.1
            weights[strategies.index("random")] = weights[strategies.index("random")] + 0.1
        
        # Chọn chiến lược
        strategy = random.choices(strategies, weights=weights, k=1)[0]
        
        self.logger.info(f"Using generation strategy: {strategy}")
        
        individual = []
        individual.extend(self.generate_constructor())
        
        # Quyết định số lượng giao dịch trong chuỗi (2-5 giao dịch)
        num_transactions = random.randint(2, min(5, settings.MAX_INDIVIDUAL_LENGTH - len(individual)))
        
        if strategy == "optimal" and self.optimal_sequences:
            # Chọn ngẫu nhiên một sequence tối ưu
            sequence_template = random.choice(self.optimal_sequences)
            self.logger.info(f"Generated sequence from optimal template: {sequence_template}")
            optimal_seq = self._generate_optimal_sequence(sequence_template)
            if optimal_seq:
                individual.extend(optimal_seq)
                return individual
            # Nếu không thành công, tiếp tục với chiến lược khác
        
        elif strategy == "vulnerability" and self.potential_vulnerabilities:
            # Chọn ngẫu nhiên một lỗ hổng để nhắm tới
            vulnerability = random.choice(self.potential_vulnerabilities)
            self.logger.info(f"Generated sequence targeting vulnerability: {vulnerability.get('type')}")
            vuln_seq = self._generate_vulnerability_targeting_sequence(vulnerability)
            if vuln_seq:
                individual.extend(vuln_seq)
                return individual
            # Nếu không thành công, tiếp tục với chiến lược khác
            
        elif strategy == "critical_path" and self.critical_paths:
            # Chọn ngẫu nhiên một critical path
            path = random.choice(self.critical_paths)
            
            if path:
                self.logger.info(f"Generated sequence following critical path: {path}")
                
                # Tạo chuỗi giao dịch từ critical path
                path_transactions = []
                for func_name in path:
                    func_hash = self._get_function_hash_by_name(func_name)
                    if func_hash and func_hash in self.interface:
                        path_transactions.extend(self.generate_individual(
                            func_hash, 
                            self.interface[func_hash]
                        ))
                
                if path_transactions:
                    individual.extend(path_transactions)
                    return individual
        
        # Nếu các chiến lược trên không thành công hoặc chiến lược là "random"
        self.logger.info("Generated random sequence (fallback)")
        
        # Tạo một chuỗi giao dịch có ý nghĩa dựa trên mối quan hệ đọc/ghi
        # Phân tích các hàm để tìm mối quan hệ đọc/ghi
        write_funcs = []  # Các hàm ghi dữ liệu
        read_funcs = []   # Các hàm đọc dữ liệu
        
        # Tìm các hàm đọc và ghi
        for func_name, func_hash in self.interface_mapper.items() if self.interface_mapper else {}:
            func_name = func_name.split("(")[0]  # Lấy tên không có tham số
            
            # Kiểm tra xem hàm này có trong phân tích dataflow không
            is_write_func = False
            is_read_func = False
            
            # Kiểm tra trong danh sách các hàm đã phân tích
            for contract_name, contract_info in self.dataflow_graph.items() if hasattr(self, 'dataflow_graph') else {}:
                if contract_name == self.contract_name and "functions" in contract_info:
                    if func_name in contract_info["functions"]:
                        func_info = contract_info["functions"][func_name]
                        if func_info.get("writes"):
                            is_write_func = True
                        if func_info.get("reads"):
                            is_read_func = True
            
            # Nếu không có thông tin dataflow, dựa vào tên hàm
            if not (is_write_func or is_read_func):
                if func_name.lower().startswith(("set", "add", "create", "update", "delete", "remove", "transfer", "mint", "burn")):
                    is_write_func = True
                elif func_name.lower().startswith(("get", "view", "is", "has", "balance", "total", "name", "symbol", "decimals")):
                    is_read_func = True
            
            if is_write_func:
                write_funcs.append((func_name, func_hash))
            if is_read_func:
                read_funcs.append((func_name, func_hash))
        
        # Tạo chuỗi giao dịch có mối quan hệ đọc/ghi
        transactions = []
        
        # Thêm một số giao dịch ghi trước
        for _ in range(min(2, num_transactions - 1)):
            if write_funcs:
                func_name, func_hash = random.choice(write_funcs)
                if func_hash in self.interface:
                    transactions.extend(self.generate_individual(
                        func_hash,
                        self.interface[func_hash]
                    ))
        
        # Thêm một số giao dịch đọc sau
        for _ in range(min(2, num_transactions - len(transactions))):
            if read_funcs:
                func_name, func_hash = random.choice(read_funcs)
                if func_hash in self.interface:
                    transactions.extend(self.generate_individual(
                        func_hash,
                        self.interface[func_hash]
                    ))
        
        # Nếu vẫn chưa đủ số lượng giao dịch, thêm các giao dịch ngẫu nhiên
        while len(transactions) < num_transactions:
            function, argument_types = self.get_random_function_with_argument_types()
            transactions.extend(self.generate_individual(function, argument_types))
        
        # Giới hạn số lượng giao dịch
        if len(transactions) > num_transactions:
            transactions = transactions[:num_transactions]
        
        individual.extend(transactions)
        
        # Hiển thị thông tin về chuỗi giao dịch đã tạo
        self.logger.info(f"Generated sequence with {len(individual)} transactions")
        for i, tx in enumerate(individual):
            func_name = "constructor" if i == 0 else tx["arguments"][0] if "arguments" in tx and len(tx["arguments"]) > 0 else "unknown"
            self.logger.debug(f"Transaction {i+1}: {func_name}")
        
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
    Hàm tiện ích để tạo RAGEnhancedGenerator
    """
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import re
import random
from typing import Dict, List, Any, Optional, Union, Tuple

# Sửa import để phù hợp với mọi phiên bản
try:
    # Thử import theo cách mới
    from google import generativeai as genai
except ImportError:
    # Fallback sang import theo cách cũ
    import google.generativeai as genai

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LLMAgent")

class LLMAgent:
    """
    Agent sử dụng LLM để sinh và tối ưu các giá trị cho fuzzing
    """
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Khởi tạo LLM Agent
        
        :param api_key: Google API Key
        :param model: Tên model LLM (mặc định: gemini-1.5-flash)
        """
        self.api_key = api_key
        self.model = model
        
        # Cấu hình API theo cách tương thích với mọi phiên bản
        genai.configure(api_key=api_key)
        self.cache = {}  # Cache kết quả để tránh gọi API nhiều lần
        
        # Khởi tạo type mapping
        self.type_mapping = {
            "bool": bool,
            "address": str,
            "string": str,
            "bytes": bytes
        }
        
        # Khởi tạo các pattern regex
        self.uint_pattern = re.compile(r"uint(\d+)")
        self.int_pattern = re.compile(r"int(\d+)")
        self.bytes_pattern = re.compile(r"bytes(\d+)")
        self.array_pattern = re.compile(r"(.*)\[\]")
        self.fixed_array_pattern = re.compile(r"(.*)\[(\d+)\]")
        
        logger.info(f"LLMAgent initialized with model: {model}")
    
    def get_argument_suggestion(self, 
                               type_str: str, 
                               function_name: str, 
                               arg_name: str, 
                               arg_index: int,
                               context: Optional[Dict] = None) -> Any:
        """
        Lấy đề xuất giá trị cho một tham số từ LLM
        
        :param type_str: Chuỗi kiểu dữ liệu (vd: "uint256", "address", "string")
        :param function_name: Tên hàm (vd: "constructor", "transfer")
        :param arg_name: Tên tham số (nếu có)
        :param arg_index: Vị trí tham số
        :param context: Bối cảnh bổ sung (audit report, phân tích...)
        :return: Giá trị được đề xuất (đã chuyển đổi sang kiểu thích hợp)
        """
        # Kiểm tra cache
        cache_key = f"{function_name}_{arg_name}_{type_str}_{arg_index}"
        logger.info(f"LLM AGENT: Generating suggestion for {cache_key}")
        if cache_key in self.cache:
            logger.info(f"LLM AGENT CACHE: Using cached value for {cache_key}: {self.cache[cache_key]}")
            return self.cache[cache_key]
            
        # Xây dựng prompt
        prompt = self._build_prompt(type_str, function_name, arg_name, arg_index, context)
        logger.debug(f"LLM PROMPT: {prompt}")
        try:
            # Gọi LLM API theo cách tương thích với mọi phiên bản
            logger.info(f"LLM AGENT: Calling API for {function_name}.{arg_name}")
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            
            # Xử lý kết quả trả về
            response_text = response.text if hasattr(response, 'text') else response.parts[0].text
            value = self._parse_llm_response(response_text, type_str)
            logger.info(f"LLM RESPONSE: {response_text}")
            # Chuyển đổi kiểu
            typed_value = self._convert_to_type(value, type_str)
            logger.info(f"LLM PARSED VALUE: {value}")
            # Cache kết quả
            self.cache[cache_key] = typed_value
            
            logger.info(f"Successfully generated value for {function_name}.{arg_name} ({type_str}): {typed_value}")
            return typed_value
            
        except Exception as e:
            logger.error(f"Error generating argument value: {e}")
            return None
    
    def _build_prompt(self, type_str: str, function_name: str, arg_name: str, arg_index: int, context: Optional[Dict]) -> str:
        """
        Xây dựng prompt cho LLM với thông tin chi tiết hơn
        """
        # Cơ bản
        prompt = f"""Generate a single optimal value for fuzzing a Solidity smart contract function parameter.

Parameter information:
- Function: {function_name}
- Parameter name: {arg_name}
- Parameter type: {type_str}
- Parameter index: {arg_index}

Goal: Generate a value that could potentially:
1. Trigger edge cases
2. Expose security vulnerabilities
3. Test boundary conditions
4. Exercise unusual code paths
"""

        # Thêm hướng dẫn dựa trên kiểu
        if type_str.startswith("uint"):
            prompt += """
For uint types:
- Consider edge cases: 0, 1, maximum value (2^256-1), power of 2 values
- Consider values that might trigger integer overflow
- Consider values near uint256 max (2^256-1)
- Consider values that could cause division by zero
- Return only the numeric value without any decorations
"""
        elif type_str.startswith("int"):
            prompt += """
For int types:
- Consider edge cases: minimum value (-2^255), -1, 0, 1, maximum value (2^255-1)
- Consider values that might trigger integer overflow or underflow
- Consider values near int256 min (-2^255) and max (2^255-1)
- Consider values that could cause division by zero
- Return only the numeric value without any decorations
"""
        elif type_str.startswith("address"):
            prompt += """
For address types:
- Give a valid 42-character Ethereum address (including '0x' prefix)
- Consider special addresses:
  * address(0) - zero address
  * msg.sender - current caller
  * address(this) - contract's own address
  * address(1) - first address
  * address(2^160-1) - last possible address
- Return only the address as a string
"""
        elif type_str.startswith("bool"):
            prompt += """
For boolean types:
- Consider both true and false values
- Consider values that might trigger short-circuit evaluation
- Return either "true" or "false" as a string
"""
        elif type_str.startswith("bytes"):
            prompt += """
For bytes types:
- Consider edge cases:
  * Empty bytes (0x)
  * Very long bytes (near gas limit)
  * Special byte patterns (all 0s, all 1s)
  * Malformed bytes
- Consider bytes that might trigger buffer overflows
- Return a hex string (with 0x prefix)
"""
        elif type_str.startswith("string"):
            prompt += """
For string types:
- Consider edge cases:
  * Empty string
  * Very long strings (near gas limit)
  * Special characters (UTF-8, emoji)
  * SQL injection patterns
  * XSS patterns
  * Buffer overflow patterns
- Return the string directly
"""
            
        # Thêm bối cảnh audit nếu có
        if context and "audit_report" in context:
            prompt += f"""
Audit report context:
{context['audit_report']}

Based on this audit report:
1. Identify potential security issues
2. Generate values that could trigger these issues
3. Focus on edge cases mentioned in the report
4. Consider any specific attack vectors described
"""

        prompt += """
Return ONLY the raw value with NO explanation, NO JSON formatting, and NO quotes (unless it's a string).
The value should be directly usable in a Solidity function call.
"""
        return prompt
    
    def _parse_llm_response(self, response_text: str, type_str: str) -> Any:
        """
        Xử lý phản hồi từ LLM và trích xuất giá trị
        """
        # Làm sạch chuỗi phản hồi
        value = response_text.strip()
        
        # Xóa dấu ngoặc kép nếu có
        if type_str != "string" and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
            
        # Xóa các ký tự không mong muốn
        value = value.replace('\n', '').replace('\r', '').strip()
        
        # Kiểm tra giá trị rỗng
        if not value:
            logger.warning(f"Empty value received from LLM for type {type_str}")
            return None
            
        # Kiểm tra giá trị hợp lệ
        if type_str.startswith("uint") or type_str.startswith("int"):
            if not value.replace("-", "").isdigit() and not value.startswith("0x"):
                logger.warning(f"Invalid numeric value received: {value}")
                return None
                
        elif type_str == "address":
            if not value.startswith("0x") or len(value) != 42:
                logger.warning(f"Invalid address format: {value}")
                return None
                
        elif type_str.startswith("bytes"):
            if not value.startswith("0x"):
                logger.warning(f"Invalid bytes format: {value}")
                return None
                
        return value
    
    def _convert_to_type(self, value: str, type_str: str) -> Any:
        """
        Chuyển đổi giá trị chuỗi sang kiểu dữ liệu thích hợp
        """
        if value is None:
            return None
            
        # Mảng
        array_match = self.array_pattern.match(type_str)
        fixed_array_match = self.fixed_array_pattern.match(type_str)
        
        if array_match or fixed_array_match:
            if array_match:
                base_type = array_match.group(1)
                # Trường hợp mảng động, bắt đầu với một phần tử
                return [self._convert_to_type(value, base_type)]
            else:
                base_type = fixed_array_match.group(1)
                size = int(fixed_array_match.group(2))
                # Tạo mảng với kích thước cố định, tất cả có cùng giá trị
                return [self._convert_to_type(value, base_type) for _ in range(size)]
        
        # Boolean
        if type_str == "bool":
            return value.lower() in ["true", "1", "yes"]
        
        # Unsigned integer
        uint_match = self.uint_pattern.match(type_str)
        if uint_match:
            try:
                num = int(value)
                # Kiểm tra giới hạn uint
                bits = int(uint_match.group(1))
                max_value = (1 << bits) - 1
                if num < 0 or num > max_value:
                    logger.warning(f"Value {num} out of range for uint{bits}")
                    return 0
                return num
            except ValueError:
                # Xử lý giá trị hex
                if value.startswith("0x"):
                    try:
                        return int(value, 16)
                    except ValueError:
                        logger.warning(f"Invalid hex value: {value}")
                        return 0
                return 0
        
        # Signed integer
        int_match = self.int_pattern.match(type_str)
        if int_match:
            try:
                num = int(value)
                # Kiểm tra giới hạn int
                bits = int(int_match.group(1))
                max_value = (1 << (bits - 1)) - 1
                min_value = -(1 << (bits - 1))
                if num < min_value or num > max_value:
                    logger.warning(f"Value {num} out of range for int{bits}")
                    return 0
                return num
            except ValueError:
                # Xử lý giá trị hex
                if value.startswith("0x"):
                    try:
                        return int(value, 16)
                    except ValueError:
                        logger.warning(f"Invalid hex value: {value}")
                        return 0
                return 0
        
        # Address
        if type_str == "address":
            # Đảm bảo địa chỉ hợp lệ
            if not value.startswith("0x"):
                value = "0x" + value
            if len(value) != 42:
                # Pad cho đủ 42 ký tự
                value = "0x" + value[2:].zfill(40)
            return value
        
        # String
        if type_str == "string":
            return value
        
        # Bytes và bytesN
        bytes_match = self.bytes_pattern.match(type_str)
        if type_str == "bytes" or bytes_match:
            if value.startswith("0x"):
                try:
                    return bytes.fromhex(value[2:])
                except ValueError:
                    logger.warning(f"Invalid hex bytes: {value}")
                    return b""
            else:
                return bytes(value, 'utf-8')
        
        # Mặc định trả về chuỗi
        return value 
    def enhance_prompt_with_audit_report(self, prompt, context, type_str, function_name):
        """
        Cải thiện prompt với các thông tin từ báo cáo audit
        """
        if context and "audit_report" in context:
            # Phân tích báo cáo audit để tìm lỗ hổng liên quan đến hàm cụ thể
            audit_content = context["audit_report"]
            
            # Trích xuất các phần liên quan đến hàm hiện tại
            function_related_info = self._extract_function_info(audit_content, function_name)
            
            # Trích xuất thông tin về kiểu dữ liệu cụ thể
            type_related_info = self._extract_type_info(audit_content, type_str)
            
            # Tìm các lỗ hổng phổ biến
            vulnerability_info = self._extract_vulnerability_info(audit_content)
            
            prompt += f"""
            Thông tin từ báo cáo audit về hàm '{function_name}':
            {function_related_info}
            
            Thông tin về kiểu dữ liệu '{type_str}':
            {type_related_info}
            
            Các lỗ hổng đã phát hiện:
            {vulnerability_info}
            
            Hãy tạo ra giá trị có thể kích hoạt lỗ hổng dựa trên thông tin từ báo cáo audit.
            """
    
        return prompt

    def _extract_function_info(self, audit_content, function_name):
        """
        Trích xuất thông tin liên quan đến hàm cụ thể từ báo cáo audit
        """
        # Sử dụng regex để tìm các đoạn liên quan đến hàm
        pattern = re.compile(rf"({function_name}[^\n.]*?)([\s\S]{{1,500}}?)(?=\n\n|\n#|\Z)", re.IGNORECASE)
        matches = pattern.findall(audit_content)
        
        if matches:
            return "\n".join(m[0] + m[1] for m in matches)
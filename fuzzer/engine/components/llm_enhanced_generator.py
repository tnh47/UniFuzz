#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import random
import re
from typing import Dict, List, Any, Optional, Union

from .generator import Generator
from .llm_agent import LLMAgent

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LLMEnhancedGenerator")

class LLMEnhancedGenerator(Generator):
    """
    Generator được cải tiến với LLM để sinh các giá trị tối ưu
    """
    
    def __init__(self, interface: Dict, 
                 bytecode: str, 
                 accounts: List[str], 
                 contract: str, 
                 api_key: str,
                 audit_report: Optional[str] = None,
                 contract_name: Optional[str] = None, 
                 sol_path: Optional[str] = None,
                 other_generators=None, 
                 interface_mapper=None):
        """
        Khởi tạo generator với LLM
        
        :param interface: Giao diện hợp đồng
        :param bytecode: Bytecode hợp đồng
        :param accounts: Danh sách tài khoản
        :param contract: Địa chỉ hợp đồng
        :param api_key: Google API Key cho LLM
        :param audit_report: Báo cáo audit (tuỳ chọn)
        :param contract_name: Tên hợp đồng
        :param sol_path: Đường dẫn tới file Solidity
        :param other_generators: Danh sách generator khác
        :param interface_mapper: Interface mapper
        """
        super().__init__(interface, bytecode, accounts, contract, 
                        other_generators=other_generators, 
                        interface_mapper=interface_mapper,
                        contract_name=contract_name, 
                        sol_path=sol_path)
        
        # Khởi tạo LLM Agent
        self.llm_agent = LLMAgent(api_key=api_key)
        self.audit_report = audit_report
        logger.info(f"LLMEnhancedGenerator initialized for contract: {contract_name}")
    
    def get_random_argument(self, type_str: str, function: str, argument_index: int) -> Any:
        """
        Override phương thức get_random_argument để sử dụng LLM
        với logging chi tiết và fallback về random generation.
        """
        # Logging đầy đủ thông tin đầu vào
        logger.info(f"LLM GENERATOR INPUT: function={function}, arg_index={argument_index}, type={type_str}")
        
        # Xây dựng context cho LLM
        context: Dict[str, Any] = {}
        if self.audit_report:
            context["audit_report"] = self.audit_report
            logger.debug(f"Using audit report for context in {function}.{argument_index}")
        
        # Nếu có file source, thêm vào context
        if hasattr(self, 'sol_path') and self.sol_path and os.path.exists(self.sol_path):
            try:
                with open(self.sol_path, 'r') as f:
                    context["contract_source"] = f.read()
                logger.debug(f"Loaded contract source for context in {function}.{argument_index}")
            except Exception as e:
                logger.warning(f"Could not read sol file {self.sol_path}: {e}")
        
        # Tham số tên
        arg_name = f"arg{argument_index}"
        
        # Cố gắng sử dụng giá trị từ audit report trước
        known_values = None
        if self.audit_report:
            known_values = self._extract_known_values_from_audit(
                function=function,
                arg_index=argument_index,
                arg_type=type_str
            )
            if known_values:
                logger.debug(f"Known audit-derived values for {function}.{arg_name}: {known_values}")
        
        # 50% chance dùng giá trị audit-derived nếu có
        if known_values and random.random() < 0.5:
            value = random.choice(known_values)
            self.add_argument_to_pool(function, argument_index, value)
            logger.info(f"LLM AUDIT VALUE: Using audit-derived value for {function}.{arg_name}: {value}")
            return value
        
        # Cố gắng lấy giá trị từ LLM agent
        try:
            logger.info(f"Attempting to generate value for {function}.{arg_name} ({type_str}) using LLM")
            llm_value = self.llm_agent.get_argument_suggestion(
                type_str=type_str,
                function_name=function,
                arg_name=arg_name,
                arg_index=argument_index,
                context=context
            )
            
            if llm_value is not None:
                self.add_argument_to_pool(function, argument_index, llm_value)
                logger.info(f"LLM SUCCESS: Generated value for {function}.{arg_name}: {llm_value}")
                return llm_value
            else:
                logger.warning(f"LLM FAILED: Returned None for {function}.{arg_name} ({type_str})")
        except Exception as e:
            logger.error(f"LLM ERROR: {e}", exc_info=True)
        
        # Fallback về random generator
        logger.info(f"FALLBACK: Using random generation for {function}.{arg_name} ({type_str})")
        try:
            random_value = super().get_random_argument(type_str, function, argument_index)
            logger.info(f"RANDOM VALUE: {random_value}")
            return random_value
        except Exception as e:
            logger.error(f"Error in random generation fallback: {e}", exc_info=True)
            # Nếu super() cũng lỗi, trả về None hoặc ném tiếp exception
            return None
        
    def _extract_known_values_from_audit(self, function, arg_index, arg_type):
        """
        Trích xuất các giá trị đã biết từ báo cáo audit
        """
        if not self.audit_report:
            return None
            
        values = []
        
        # Tìm các giá trị số dựa trên kiểu
        if arg_type.startswith("uint") or arg_type.startswith("int"):
            # Tìm các số trong báo cáo audit gần với tên hàm
            pattern = re.compile(rf"{function}[^.]*?(\d+)")
            matches = pattern.findall(self.audit_report)
            if matches:
                for m in matches:
                    try:
                        values.append(int(m))
                    except ValueError:
                        pass
        
        # Tìm các địa chỉ
        if arg_type.startswith("address"):
            pattern = re.compile(r"0x[a-fA-F0-9]{40}")
            matches = pattern.findall(self.audit_report)
            values.extend(matches)
        
        return values if values else None
    def generate_constructor(self) -> List[Dict[str, Any]]:
        """
        Override phương thức generate_constructor để sử dụng LLM với logging chi tiết
        và fallback về random generator khi LLM không trả về giá trị.
        """
        individual = []

        if "constructor" in self.interface and self.bytecode:
            logger.info("LLM CONSTRUCTOR: Generating constructor with LLM enhancement")
            arguments = ["constructor"]

            # Tạo metadata cho tất cả tham số để gửi cho LLM
            constructor_params = []
            for index, arg_type in enumerate(self.interface["constructor"]):
                constructor_params.append({
                    "index": index,
                    "type": arg_type,
                    "name": f"arg{index}"
                })
            logger.debug(f"LLM CONSTRUCTOR ANALYSIS: Params metadata = {constructor_params}")

            # Xây dựng context phong phú cho LLM
            context: Dict[str, Any] = {
                "audit_report": self.audit_report,
                "contract_name": self.contract_name
            }
            if self.sol_path and os.path.exists(self.sol_path):
                try:
                    with open(self.sol_path, "r") as f:
                        context["contract_source"] = f.read()
                    logger.debug("Loaded contract source into context for constructor generation")
                except Exception as e:
                    logger.warning(f"Could not read Solidity file {self.sol_path}: {e}")

            # Gọi LLM để phân tích constructor
            try:
                constructor_analysis = self.llm_agent.analyze_constructor(
                    constructor_params=constructor_params,
                    context=context
                )
                logger.info(f"LLM CONSTRUCTOR RESULT: {constructor_analysis}")
            except Exception as e:
                logger.error(f"LLM ERROR during constructor analysis: {e}", exc_info=True)
                constructor_analysis = None

            # Sinh giá trị cho từng tham số
            for index, arg_type in enumerate(self.interface["constructor"]):
                arg_name = f"arg{index}"
                use_llm = False

                # Nếu có kết quả phân tích từ LLM và giá trị hợp lệ, dùng nó
                if constructor_analysis and index < len(constructor_analysis):
                    suggested = constructor_analysis[index].get("value")
                    if suggested is not None:
                        logger.info(f"LLM CONSTRUCTOR USING: {arg_name} = {suggested}")
                        arguments.append(suggested)
                        use_llm = True

                # Fallback về random nếu LLM không trả về giá trị
                if not use_llm:
                    logger.info(f"LLM CONSTRUCTOR FALLBACK: Generating random value for {arg_name} ({arg_type})")
                    try:
                        rnd = self.get_random_argument(arg_type, "constructor", index)
                        logger.info(f"RANDOM CONSTRUCTOR VALUE: {arg_name} = {rnd}")
                        arguments.append(rnd)
                    except Exception as e:
                        logger.error(f"Error generating random constructor argument for {arg_name}: {e}", exc_info=True)
                        arguments.append(None)

            logger.debug(f"FINAL CONSTRUCTOR ARGUMENTS: {arguments}")

            # Tạo individual record cho constructor
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

def create_llm_enhanced_generator(
        interface: Dict, 
        bytecode: str,
        accounts: List[str],
        contract: str,
        api_key: str,
        audit_report: Optional[str] = None,
        contract_name: Optional[str] = None,
        sol_path: Optional[str] = None,
        other_generators=None,
        interface_mapper=None) -> LLMEnhancedGenerator:
    """
    Helper function để tạo LLMEnhancedGenerator
    """
    return LLMEnhancedGenerator(
        interface=interface,
        bytecode=bytecode,
        accounts=accounts,
        contract=contract,
        api_key=api_key,
        audit_report=audit_report,
        contract_name=contract_name,
        sol_path=sol_path,
        other_generators=other_generators,
        interface_mapper=interface_mapper
    )
 
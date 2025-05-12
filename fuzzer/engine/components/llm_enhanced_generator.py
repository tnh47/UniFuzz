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
    Generator được cải tiến với RAG để sinh các giá trị tối ưu
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
        Khởi tạo generator với RAG
        
        :param interface: Giao diện hợp đồng
        :param bytecode: Bytecode hợp đồng
        :param accounts: Danh sách tài khoản
        :param contract: Địa chỉ hợp đồng
        :param api_key: Google API Key cho RAG
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
        
        # Khởi tạo LLM Agent với RAG
        self.llm_agent = LLMAgent(api_key=api_key)
        self.audit_report = audit_report
        logger.info(f"LLMEnhancedGenerator initialized for contract: {contract_name}")
    
    def get_random_argument(self, type_str: str, function: str, argument_index: int) -> Any:
        """
        Override phương thức get_random_argument để sử dụng RAG
        với logging chi tiết và fallback về random generation.
        """
        logger.info(f"RAG GENERATOR INPUT: function={function}, arg_index={argument_index}, type={type_str}")
        
        # Xây dựng context cho RAG
        context: Dict[str, Any] = {}
        if self.audit_report:
            context["audit_report"] = self.audit_report
            logger.debug(f"Using audit report for context in {function}.{argument_index}")
        
        if hasattr(self, 'sol_path') and self.sol_path and os.path.exists(self.sol_path):
            try:
                with open(self.sol_path, 'r') as f:
                    context["contract_source"] = f.read()
                logger.debug(f"Loaded contract source for context in {function}.{argument_index}")
            except Exception as e:
                logger.warning(f"Could not read sol file {self.sol_path}: {e}")
        
        arg_name = f"arg{argument_index}"
        
        # Cố gắng lấy giá trị từ RAG
        try:
            logger.info(f"Attempting to generate value for {function}.{arg_name} ({type_str}) using RAG")
            rag_value = self.llm_agent.get_argument_suggestion(
                type_str=type_str,
                function_name=function,
                arg_name=arg_name,
                arg_index=argument_index,
                context=context
            )
            
            if rag_value is not None:
                self.add_argument_to_pool(function, argument_index, rag_value)
                logger.info(f"RAG SUCCESS: Generated value for {function}.{arg_name}: {rag_value}")
                return rag_value
            else:
                logger.warning(f"RAG FAILED: Returned None for {function}.{arg_name} ({type_str})")
        except Exception as e:
            logger.error(f"RAG ERROR: {e}", exc_info=True)
        
        # Fallback về random generator
        logger.info(f"FALLBACK: Using random generation for {function}.{arg_name} ({type_str})")
        try:
            random_value = super().get_random_argument(type_str, function, argument_index)
            logger.info(f"RANDOM VALUE: {random_value}")
            return random_value
        except Exception as e:
            logger.error(f"Error in random generation fallback: {e}", exc_info=True)
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
    def generate_individual(self, function, argument_types, default_value=False):
        """Sinh một giao dịch với tham số và hiển thị chi tiết"""
        individual = []
        arguments = [function]  # Function selector là tham số đầu tiên
        args_values = []
        
        # Sinh các tham số
        for index in range(len(argument_types)):
            arg_value = self.get_random_argument(argument_types[index], function, index)
            arguments.append(arg_value)
            args_values.append(f"{argument_types[index]}:{arg_value}")
        
        # Lấy tên hàm nếu có
        function_name = "unknown"
        if self.interface_mapper:
            for fname, fhash in self.interface_mapper.items():
                if fhash == function:
                    function_name = fname
                    break
        
        # Tạo transaction
        account = self.get_random_account(function)
        amount = self.get_random_amount(function)
        gas_limit = self.get_random_gaslimit(function)
        
        # In thông tin transaction
        print(f"-----------------------------------------------------")
        print(f"Transaction - {function_name}:")
        print(f"-----------------------------------------------------")
        print(f"From:      {account}")
        print(f"To:        {self.contract}")
        print(f"Value:     {amount} Wei")
        print(f"Gas Limit: {gas_limit}")
        print(f"Input:     {function}{' '.join([str(arg) for arg in arguments[1:]])}")
        print(f"-----------------------------------------------------")
        
        # Log với logger
        logger.info(f"Transaction - {function_name}:")
        logger.info(f"From: {account}, To: {self.contract}")
        logger.info(f"Args: {', '.join(args_values)}")
        
        # Thêm vào individual
        individual.append({
            "account": account,
            "contract": self.contract,
            "amount": amount,
            "arguments": arguments,
            "blocknumber": self.get_random_blocknumber(function),
            "timestamp": self.get_random_timestamp(function),
            "gaslimit": gas_limit,
            "call_return": dict(),
            "extcodesize": dict(),
            "returndatasize": dict()
        })
        
        # Phần còn lại của hàm không thay đổi
        address, call_return_value = self.get_random_callresult_and_address(function)
        individual[-1]["call_return"] = {address: call_return_value}
        
        address, extcodesize_value = self.get_random_extcodesize_and_address(function)
        individual[-1]["extcodesize"] = {address: extcodesize_value}
        
        address, value = self.get_random_returndatasize_and_address(function)
        individual[-1]["returndatasize"] = {address: value}
        
        return individual


    def generate_constructor(self) -> List[Dict[str, Any]]:
        """
        Override phương thức generate_constructor để sử dụng RAG với logging chi tiết
        và fallback về random generator khi RAG không trả về giá trị.
        """
        individual = []

        if "constructor" in self.interface and self.bytecode:
            logger.info("RAG CONSTRUCTOR: Generating constructor with RAG enhancement")
            arguments = ["constructor"]

            # Tạo metadata cho tất cả tham số để gửi cho RAG
            constructor_params = []
            for index, arg_type in enumerate(self.interface["constructor"]):
                constructor_params.append({
                    "index": index,
                    "type": arg_type,
                    "name": f"arg{index}"
                })
            logger.debug(f"RAG CONSTRUCTOR ANALYSIS: Params metadata = {constructor_params}")

            # Xây dựng context phong phú cho RAG
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

            # Sinh giá trị cho từng tham số
            for index, arg_type in enumerate(self.interface["constructor"]):
                arg_name = f"arg{index}"
                logger.info(f"RAG CONSTRUCTOR: Generating value for {arg_name} ({arg_type})")
                
                try:
                    rag_value = self.llm_agent.get_argument_suggestion(
                        type_str=arg_type,
                        function_name="constructor",
                        arg_name=arg_name,
                        arg_index=index,
                        context=context
                    )
                    if rag_value is not None:
                        logger.info(f"RAG CONSTRUCTOR USING: {arg_name} = {rag_value}")
                        arguments.append(rag_value)
                        continue
                except Exception as e:
                    logger.error(f"RAG ERROR for {arg_name}: {e}", exc_info=True)
                
                # Fallback về random nếu RAG không trả về giá trị
                logger.info(f"RAG CONSTRUCTOR FALLBACK: Generating random value for {arg_name} ({arg_type})")
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
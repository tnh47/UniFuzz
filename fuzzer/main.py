#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
from datetime import datetime

import solcx
import random
import argparse

from eth_utils import encode_hex, to_canonical_address
from z3 import Solver
from eth.db.account import Account

from evm import InstrumentedEVM
from detectors import DetectorExecutor
from engine import EvolutionaryFuzzingEngine
from engine.components.generator import Generator
from engine.components import Individual, Population
from engine.analysis import SymbolicTaintAnalyzer
from engine.analysis import ExecutionTraceAnalyzer
from engine.environment import FuzzingEnvironment
from engine.operators import LinearRankingSelection
from engine.operators import DataDependencyLinearRankingSelection
from engine.operators import Crossover
from engine.operators import DataDependencyCrossover
from engine.operators import Mutation
from engine.fitness import fitness_function
from fuzzer.utils.transaction_seq_utils import check_cross_init, gen_trans, init_func

# 获取根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将根目录添加到path中
sys.path.append(BASE_DIR)

# Imports sau khi đã thêm BASE_DIR vào sys.path
from fuzzer.utils import settings
from utils.source_map import SourceMap
from utils.utils import initialize_logger, compile, get_interface_from_abi, get_pcs_and_jumpis, \
    get_function_signature_mapping
from utils.control_flow_graph import ControlFlowGraph

# Import RAG components safely
try:
    from engine.analyzers.dataflow_analyzer import SmartContractAnalyzer
    from engine.components.generator import Generator
    # Import RAG components
    from engine.components.rag_enhanced_generator import RAGEnhancedGenerator, create_rag_enhanced_generator
    from engine.components.rag_enhanced_population import RAGEnhancedPopulation
    from engine.components.llm_enhanced_generator import LLMEnhancedGenerator, create_llm_enhanced_generator
    rag_available = True
except ImportError as e:
    import logging
    logging.warning(f"Could not import RAG components: {e}")
    logging.warning("Will use standard generator if RAG is requested")
    rag_available = False
    # Để xử lý isinstance(), định nghĩa RAGEnhancedGenerator như một class trống
    class RAGEnhancedGenerator:
        pass


class Fuzzer:
    def __init__(self, contract_name, abi, deployment_bytecode, runtime_bytecode, test_instrumented_evm,
                 blockchain_state, solver, args, seed, source_map=None, whole_compile_info=None):
        global logger

        logger = initialize_logger("Fuzzer  ")
        logger.title("Fuzzing contract %s", contract_name)

        cfg = ControlFlowGraph()
        cfg.build(runtime_bytecode, settings.EVM_VERSION)

        self.contract_name = contract_name
        self.interface, self.interface_mapper = get_interface_from_abi(abi)
        self.deployement_bytecode = deployment_bytecode
        self.blockchain_state = blockchain_state
        self.instrumented_evm = test_instrumented_evm
        self.solver = solver
        self.args = args
        # 该合约依赖的其他合约
        self.depend_contracts = args.depend_contracts
        self.whole_compile_info = whole_compile_info

        # Get some overall metric on the code
        self.overall_pcs, self.overall_jumpis = get_pcs_and_jumpis(
            runtime_bytecode)

        # Initialize results
        self.results = {"errors": {}}

        # Initialize fuzzing environment
        self.env = FuzzingEnvironment(instrumented_evm=self.instrumented_evm,
                                      contract_name=self.contract_name,
                                      solver=self.solver,
                                      results=self.results,
                                      symbolic_taint_analyzer=SymbolicTaintAnalyzer(),
                                      detector_executor=DetectorExecutor(source_map,
                                                                         get_function_signature_mapping(abi)),
                                      interface=self.interface,
                                      overall_pcs=self.overall_pcs,
                                      overall_jumpis=self.overall_jumpis,
                                      len_overall_pcs_with_children=0,
                                      other_contracts=list(),
                                      args=args,
                                      seed=seed,
                                      cfg=cfg,
                                      abi=abi)
        init_func(args.source)  # 初始化分析跨合约序列
        assert check_cross_init(), "跨合约初始化失败"  # 检查是否初始化成功
        print("跨合约初始化成功......")

    def deploy_depend_contracts(self):
        generators = []
        if self.whole_compile_info is None:
            logger.error("没有找到编译信息, 退出程序!")
            sys.exit(-1)
            
        if self.args.source and len(self.depend_contracts) != 0:  # Nếu cần deploy dependent contracts
            for contract_name in self.depend_contracts:
                if contract_name == self.contract_name:
                    logger.error(contract_name + " is the same as the contract to be fuzzed!")
                    sys.exit(-1)

                contract = self.whole_compile_info[contract_name]
                if contract['abi'] and contract['evm']['bytecode']['object']:
                    # Lấy interface và bytecode của contract
                    interface, interface_mapper = get_interface_from_abi(contract['abi'])
                    deployement_bytecode = contract['evm']['bytecode']['object']
                    
                    # Xóa constructor từ interface nếu có
                    if "constructor" in interface:
                        del interface['constructor']
                    
                    # Deploy contract phụ thuộc
                    if "constructor" not in interface:
                        # Tạo deploy_args phù hợp với hợp đồng phụ thuộc
                        deploy_args = []
                        
                        # Kiểm tra xem hợp đồng có cần tham số constructor đặc biệt không
                        if contract_name == "SafeMath" or contract_name == "BasicToken" or contract_name == "ERC20Basic":
                            # Những hợp đồng này thường không cần tham số
                            logger.info(f"No constructor args needed for {contract_name}")
                        elif contract_name == "ERC20" or contract_name == "StandardToken":
                            # Nếu cần tham số, phải đúng định dạng [name, type, value]
                            deploy_args = [
                                "name_", "string", f"{contract_name} Token",
                                "symbol_", "string", f"{contract_name[0:3]}"
                            ]
                            logger.info(f"Using constructor args for {contract_name}: {deploy_args}")
                        
                        # Sử dụng tài khoản đầu tiên để deploy
                        result = self.instrumented_evm.deploy_contract(
                            self.instrumented_evm.accounts[0],
                            deployement_bytecode,
                            deploy_args=deploy_args,
                            deploy_mode=settings.CROSS_INIT_MODE
                        )
                        
                        if result.is_error:
                            logger.error(f"Problem while deploying dependent contract {contract_name} using account {self.instrumented_evm.accounts[0]}. Error message: {result._error}")
                            # Không exit nếu deploy thất bại, chỉ log lỗi
                        else:
                            # Lưu địa chỉ contract đã deploy
                            contract_address = encode_hex(result.msg.storage_address)
                            self.instrumented_evm.accounts.append(contract_address)
                            self.env.nr_of_transactions += 1
                            logger.info(f"Dependent contract {contract_name} deployed at {contract_address}, by {self.instrumented_evm.accounts[0]}")
                            
                            # Lưu thông tin deploy
                            settings.TRANS_INFO[contract_name] = contract_address
                            settings.DEPLOYED_CONTRACT_ADDRESS[contract_name] = contract_address
                            
                            # Thêm contract vào other_contracts
                            self.env.other_contracts.append(to_canonical_address(contract_address))
                            cc, _ = get_pcs_and_jumpis(
                                self.instrumented_evm.get_code(to_canonical_address(contract_address)).hex()
                            )
                            self.env.len_overall_pcs_with_children += len(cc)
                            
                            # Tạo generator cho contract này
                            generator = Generator(
                                interface=interface, 
                                bytecode=deployement_bytecode,
                                accounts=self.instrumented_evm.accounts, 
                                contract=contract_address,
                                interface_mapper=interface_mapper, 
                                contract_name=contract_name,
                                sol_path=self.args.source
                            )
                            generators.append(generator)
                            
        return generators

    def run(self):
        settings.TRANS_INFO["contract_name"] = self.contract_name
        settings.TRANS_INFO["source_path"] = self.args.source
        settings.TRANS_INFO["start_time"] = str(datetime.now())
        settings.MAIN_CONTRACT_NAME = self.contract_name
        
        # Tạo fake accounts (từ CrossFuzz gốc)
        self.instrumented_evm.create_fake_accounts()
        
        # Triển khai các hợp đồng phụ thuộc
        if self.args.cross_contract == 1:  # Nếu bật chế độ cross-contract
            generators = self.deploy_depend_contracts() 
        else:
            generators = []

        contract_address = None
        if self.args.source:
            for transaction in self.blockchain_state:  # Nếu khối có giao dịch ban đầu, thực thi chúng
                if transaction['from'].lower() not in self.instrumented_evm.accounts:
                    self.instrumented_evm.accounts.append(
                        self.instrumented_evm.create_fake_account(transaction['from']))

                if not transaction['to']:
                    result = self.instrumented_evm.deploy_contract(transaction['from'], transaction['input'],
                                                                int(transaction['value']), int(
                                                                    transaction['gas']),
                                                                int(transaction['gasPrice']))
                    if result.is_error:
                        logger.error("Problem while deploying contract %s using account %s. Error message: %s",
                                    self.contract_name, transaction['from'], result._error)
                        sys.exit(-2)
                    else:
                        contract_address = encode_hex(result.msg.storage_address)
                        self.instrumented_evm.accounts.append(contract_address)
                        self.env.nr_of_transactions += 1
                        logger.debug("Contract deployed at %s", contract_address)
                        self.env.other_contracts.append(
                            to_canonical_address(contract_address))
                        cc, _ = get_pcs_and_jumpis(
                            self.instrumented_evm.get_code(to_canonical_address(contract_address)).hex())
                        self.env.len_overall_pcs_with_children += len(cc)
                else:
                    input = {}
                    input["block"] = {}
                    input["transaction"] = {
                        "from": transaction["from"],
                        "to": transaction["to"],
                        "gaslimit": int(transaction["gas"]),
                        "value": int(transaction["value"]),
                        "data": transaction["input"]
                    }
                    input["global_state"] = {}
                    out = self.instrumented_evm.deploy_transaction(
                        input, int(transaction["gasPrice"]))

            if "constructor" in self.interface:
                del self.interface["constructor"]

            # Nếu contract chưa được deploy, deploy nó
            if not contract_address:
                if "constructor" not in self.interface:
                    # Sử dụng cách triển khai từ CrossFuzz gốc
                    deploy_args = self.args.constructor_args
                    
                    # Đảm bảo deploy_args có định dạng đúng [name, type, value, name, type, value, ...]
                    if deploy_args and deploy_args[0] == "auto":
                        # Tự động tạo deploy_args với giá trị mặc định
                        deploy_args = []
                        
                        # Tìm contract name để tạo params phù hợp
                        if self.contract_name == "ABE":
                            # ABE constructor cần 2 tham số: name_ và symbol_
                            deploy_args = [
                                "name_", "string", "Advanced Blockchain Token",
                                "symbol_", "string", "ABE"
                            ]
                            logger.info(f"Auto-generated constructor args for {self.contract_name}: {deploy_args}")
                        else:
                            # Constructor thường không cần tham số
                            logger.info(f"No specific constructor args needed for {self.contract_name}")
                    elif deploy_args and len(deploy_args) > 0:
                        # Kiểm tra xem deploy_args có đủ bội số của 3 không
                        if len(deploy_args) % 3 != 0:
                            logger.warning(f"Constructor args not in multiples of 3: {deploy_args}")
                            logger.warning("Format should be: [name1, type1, value1, name2, type2, value2, ...]")
                            # Điều chỉnh để đảm bảo đủ bội số của 3
                            while len(deploy_args) % 3 != 0:
                                deploy_args.append("YA_DO_NOT_KNOW")  # Thêm giá trị mặc định
                    
                    logger.info(f"Deploying contract {self.contract_name} with deploy_args: {deploy_args}")
                    result = self.instrumented_evm.deploy_contract(
                        self.instrumented_evm.accounts[0],
                        self.deployement_bytecode,
                        deploy_args=deploy_args,
                        deploy_mode=settings.CROSS_INIT_MODE
                    )
                    
                    if result.is_error:
                        logger.error("Problem while deploying contract %s using account %s. Error message: %s",
                                    self.contract_name, self.instrumented_evm.accounts[0], result._error)
                        sys.exit(-2)
                    else:
                        contract_address = encode_hex(result.msg.storage_address)
                        self.instrumented_evm.accounts.append(contract_address)
                        self.env.nr_of_transactions += 1
                        logger.info("Contract deployed at %s", contract_address)
                        # Lưu thông tin về contract đã deploy
                        settings.TRANS_INFO[self.contract_name] = contract_address
                        settings.DEPLOYED_CONTRACT_ADDRESS[self.contract_name] = contract_address
            
            # Xóa địa chỉ contract khỏi accounts để tránh gửi transactions từ hợp đồng
            if contract_address in self.instrumented_evm.accounts:
                self.instrumented_evm.accounts.remove(contract_address)
            
            # Cập nhật overall_pcs và overall_jumpis từ deployed bytecode
            self.env.overall_pcs, self.env.overall_jumpis = get_pcs_and_jumpis(
                self.instrumented_evm.get_code(to_canonical_address(contract_address)).hex())
                
        elif self.args.abi:
            contract_address = self.args.contract

        self.instrumented_evm.create_snapshot()  # Tạo snapshot sau khi deploy tất cả contracts

        # Thêm phân tích dataflow nếu sử dụng RAG
        analysis_result = None
        if hasattr(self.args, 'use_rag') and self.args.use_rag and self.args.api_key and self.args.source:
            logger.info("Running dataflow analysis for RAG-enhanced fuzzing")
            try:
                analyzer = SmartContractAnalyzer(self.args.source, self.args.api_key, self.args.solc_path_cross)
                analysis_data = analyzer.analyze()
                analysis_result = analysis_data["analysis_result"]
                logger.info(f"Dataflow analysis complete: found {len(analysis_result.get('critical_paths', []))} critical paths, {len(analysis_result.get('test_sequences', []))} test sequences, and {len(analysis_result.get('vulnerabilities', []))} potential vulnerabilities")
            except Exception as e:
                logger.error(f"Error running dataflow analysis: {e}")
                logger.warning("Will continue with standard fuzzing")
                analysis_result = None

        # Tạo generator (thông thường, LLM hoặc RAG + Dataflow)
        rag_generator = None  # Biến để theo dõi RAG generator nếu được sử dụng
        
        if hasattr(self.args, 'use_rag') and self.args.use_rag and self.args.api_key and analysis_result:
            if rag_available:
                logger.info("Using RAG-enhanced generator with dataflow analysis")
                try:
                    generator = create_rag_enhanced_generator(
                        interface=self.interface,
                        bytecode=self.deployement_bytecode,
                        accounts=self.instrumented_evm.accounts,
                        contract=contract_address,
                        api_key=self.args.api_key,
                        analysis_result=analysis_result,
                        contract_name=self.contract_name,
                        sol_path=self.args.source,
                        other_generators=generators,
                        interface_mapper=self.interface_mapper
                    )
                    logger.info("RAG-enhanced generator created successfully")
                    rag_generator = generator  # Lưu lại để sử dụng later
                except Exception as e:
                    logger.error(f"Error creating RAG-enhanced generator: {e}")
                    logger.warning("Falling back to standard generator")
                    generator = Generator(
                        interface=self.interface,
                        bytecode=self.deployement_bytecode,
                        accounts=self.instrumented_evm.accounts,
                        contract=contract_address,
                        other_generators=generators,
                        interface_mapper=self.interface_mapper,
                        contract_name=self.contract_name,
                        sol_path=self.args.source
                    )
            else:
                logger.warning("RAG components not available, falling back to standard generator")
                generator = Generator(
                    interface=self.interface,
                    bytecode=self.deployement_bytecode,
                    accounts=self.instrumented_evm.accounts,
                    contract=contract_address,
                    other_generators=generators,
                    interface_mapper=self.interface_mapper,
                    contract_name=self.contract_name,
                    sol_path=self.args.source
                )
        else:
            # Sử dụng Generator thông thường
            generator = Generator(
                interface=self.interface,
                bytecode=self.deployement_bytecode,
                accounts=self.instrumented_evm.accounts,
                contract=contract_address,
                other_generators=generators,
                interface_mapper=self.interface_mapper,
                contract_name=self.contract_name,
                sol_path=self.args.source
            )

        # update the generator with the interface of the other contracts
        all_generators = [generator] + generators
        for gen in generators:
            gen.update_other_generators(all_generators, generator.total_interface_mapper)

        # Tạo population (RAGEnhancedPopulation nếu sử dụng RAG)
        size = 2 * len(self.interface)
        if hasattr(self.args, 'use_rag') and self.args.use_rag and self.args.api_key and analysis_result and rag_available:
            logger.info("Using RAG-enhanced population")
            try:
                population = RAGEnhancedPopulation(
                    indv_template=Individual(generator=generator, other_generators=generators),
                    indv_generator=generator,
                    size=settings.POPULATION_SIZE if settings.POPULATION_SIZE else size,
                    other_generators=generators).init(init_seed=False)
            except Exception as e:
                logger.error(f"Error creating RAG-enhanced population: {e}")
                logger.warning("Falling back to standard population")
                population = Population(
                    indv_template=Individual(generator=generator, other_generators=generators),
                    indv_generator=generator,
                    size=settings.POPULATION_SIZE if settings.POPULATION_SIZE else size,
                    other_generators=generators).init(init_seed=False)
        else:
            # Sử dụng Population thông thường
            population = Population(
                indv_template=Individual(generator=generator, other_generators=generators),
                             indv_generator=generator,
                             size=settings.POPULATION_SIZE if settings.POPULATION_SIZE else size,
                             other_generators=generators).init(init_seed=False)
        
        # Create genetic operators
        if self.args.data_dependency:
            selection = DataDependencyLinearRankingSelection(env=self.env)  # 基于Read After Write关系的种子选择
            crossover = DataDependencyCrossover(pc=settings.PROBABILITY_CROSSOVER, env=self.env)  # 基于数据流的交叉策略
            mutation = Mutation(pm=settings.PROBABILITY_MUTATION)
        else:
            selection = LinearRankingSelection()
            crossover = Crossover(pc=settings.PROBABILITY_CROSSOVER)
            mutation = Mutation(pm=settings.PROBABILITY_MUTATION)

        # Create and run our evolutionary fuzzing engine
        engine = EvolutionaryFuzzingEngine(population=population, selection=selection, crossover=crossover,
                                           mutation=mutation,
                                           mapping=get_function_signature_mapping(self.env.abi))
        engine.fitness_register(lambda x: fitness_function(x, self.env))  # 计算x的适应度, x是individual
        engine.analysis.append(ExecutionTraceAnalyzer(self.env))  # 注册了执行器

        self.env.execution_begin = time.time()
        self.env.population = population
        settings.GLOBAL_ENV = self.env

        engine.run(ng=settings.GENERATIONS)

        # Tính toán kết quả và hoàn thiện quá trình fuzzing
        execution_time = time.time() - self.env.execution_begin
        
        # Chuẩn bị kết quả chi tiết
        detailed_results = {
            "code_coverage": self.env.code_coverage * 100,
            "branch_coverage": self.env.branch_coverage * 100,
            "total_transactions": self.env.nr_of_transactions,
            "unique_transactions": len(self.env.unique_transactions),
            "execution_time": execution_time,
            "memory_usage": self.env.memory_consumption,
            "errors": self.results["errors"],
        }
        
        # Nếu sử dụng RAG, hoàn thiện quá trình với tổng kết và lưu kết quả
        if rag_generator:
            try:
                # Định nghĩa thư mục kết quả
                output_dir = "./fuzzing_results"
                if self.args.results:
                    output_dir = os.path.dirname(self.args.results)
                
                # Lưu kết quả chi tiết
                rag_generator.finalize_fuzzing(detailed_results, output_dir)
                
                # Log khi generator đã tổng kết xong
                logger.info("RAG-enhanced fuzzing results saved to %s", output_dir)
            except Exception as e:
                logger.error(f"Error finalizing RAG-enhanced fuzzing: {e}")

        if self.env.args.cfg:
            if self.env.args.source:
                self.env.cfg.save_control_flow_graph(
                    os.path.splitext(self.env.args.source)[0] + '-' + self.contract_name, 'pdf')
            elif self.env.args.abi:
                self.env.cfg.save_control_flow_graph(
                    os.path.join(os.path.dirname(self.env.args.abi), self.contract_name), 'pdf')

        self.instrumented_evm.reset()
        settings.TRANS_INFO["end_time"] = str(datetime.now())


def main():
    args = launch_argument_parser()

    logger = initialize_logger("Main    ")

    # Check if contract has already been analyzed
    if args.results and os.path.exists(args.results):
        os.remove(args.results)
        logger.info("Contract " + str(args.source) + " has already been analyzed: " + str(args.results))
        logger.info(f"原始的测试输出文件{args.results}已被删除")

    # Initializing random
    if args.seed:
        seed = args.seed
        if not "PYTHONHASHSEED" in os.environ:
            logger.debug("Please set PYTHONHASHSEED to '1' for Python's hash function to behave deterministically.")
    else:
        seed = random.random()
    random.seed(seed)
    logger.title("Initializing seed to %s", seed)

    # Initialize EVM
    instrumented_evm = InstrumentedEVM(settings.RPC_HOST, settings.RPC_PORT)
    instrumented_evm.set_vm_by_name(settings.EVM_VERSION)

    # Create Z3 solver instance
    solver = Solver()
    solver.set("timeout", settings.SOLVER_TIMEOUT)

    # Parse blockchain state if provided
    blockchain_state = []
    if args.blockchain_state:
        if args.blockchain_state.endswith(".json"):
            with open(args.blockchain_state) as json_file:
                for line in json_file.readlines():
                    blockchain_state.append(json.loads(line))
        elif args.blockchain_state.isnumeric():
            settings.BLOCK_HEIGHT = int(args.blockchain_state)
            instrumented_evm.set_vm(settings.BLOCK_HEIGHT)
        else:
            logger.error("Unsupported input file: " + args.blockchain_state)
            sys.exit(-1)

    # Compile source code to get deployment bytecode, runtime bytecode and ABI
    if args.source:
        if args.source.endswith(".sol"):
            compiler_output = compile(args.solc_version, settings.EVM_VERSION, args.source)
            if not compiler_output:
                logger.error("No compiler output for: " + args.source)
                sys.exit(-1)
            for contract_name, contract in compiler_output['contracts'][args.source].items():
                if args.contract and contract_name != args.contract:
                    continue
                if contract['abi'] and contract['evm']['bytecode']['object'] and contract['evm']['deployedBytecode'][
                    'object']:
                    source_map = SourceMap(':'.join([args.source, contract_name]), compiler_output)
                    Fuzzer(contract_name, contract["abi"], contract['evm']['bytecode']['object'],
                           contract['evm']['deployedBytecode']['object'], instrumented_evm, blockchain_state, solver,
                           args, seed, source_map, compiler_output['contracts'][args.source]).run()
        else:
            logger.error("Unsupported input file: " + args.source)
            sys.exit(-1)

    if args.abi:
        with open(args.abi) as json_file:
            abi = json.load(json_file)
            runtime_bytecode = instrumented_evm.get_code(to_canonical_address(args.contract)).hex()
            Fuzzer(args.contract, abi, None, runtime_bytecode, instrumented_evm, solver, args,
                   seed).run()


def launch_argument_parser():
    parser = argparse.ArgumentParser()

    # Contract parameters
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("-s", "--source", type=str,
                        help="Solidity smart contract source code file (.sol).")
    group1.add_argument("-a", "--abi", type=str,
                        help="Smart contract ABI file (.json).")

    # group2 = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("-c", "--contract", type=str,
                        help="Contract name to be fuzzed (if Solidity source code file provided) or blockchain contract address (if ABI file provided).")

    parser.add_argument("-b", "--blockchain-state", type=str,
                        help="Initialize fuzzer with a blockchain state by providing a JSON file (if Solidity source code file provided) or a block number (if ABI file provided).")

    # Compiler parameters
    parser.add_argument("--solc", help="Solidity compiler version (default '" + str(
        solcx.get_solc_version()) + "'). Installed compiler versions: " + str(
        solcx.get_installed_solc_versions()) + ".",
                        action="store", dest="solc_version", type=str)
    parser.add_argument("--evm", help="Ethereum VM (default '" + str(
        settings.EVM_VERSION) + "'). Available VM's: 'homestead', 'byzantium' or 'petersburg'.", action="store",
                        dest="evm_version", type=str)

    # Evolutionary parameters
    group3 = parser.add_mutually_exclusive_group(required=False)
    group3.add_argument("-g", "--generations",
                        help="Number of generations (default " + str(settings.GENERATIONS) + ").", action="store",
                        dest="generations", type=int)
    group3.add_argument("-t", "--timeout",
                        help="Number of seconds for fuzzer to stop.", action="store",
                        dest="global_timeout", type=int)
    parser.add_argument("-n", "--population-size",
                        help="Size of the population.", action="store",
                        dest="population_size", type=int)
    parser.add_argument("-pc", "--probability-crossover",
                        help="Size of the population.", action="store",
                        dest="probability_crossover", type=float)
    parser.add_argument("-pm", "--probability-mutation",
                        help="Size of the population.", action="store",
                        dest="probability_mutation", type=float)

    # Miscellaneous parameters
    parser.add_argument("-r", "--results", type=str, help="Folder or JSON file where results should be stored.")
    parser.add_argument("--seed", type=float, help="Initialize the random number generator with a given seed.")
    parser.add_argument("--cfg", help="Build control-flow graph and highlight code coverage.", action="store_true")
    parser.add_argument("--rpc-host", help="Ethereum client RPC hostname.", action="store", dest="rpc_host", type=str)
    parser.add_argument("--rpc-port", help="Ethereum client RPC port.", action="store", dest="rpc_port", type=int)
    
    # LLM & RAG parameters
    parser.add_argument("--api-key", help="Google API Key for LLM-enhanced fuzzing.", action="store", dest="api_key", type=str)
    parser.add_argument("--use-llm", help="Enable LLM-enhanced fuzzing (requires --api-key).", action="store_true", dest="use_llm")
    parser.add_argument("--use-rag", help="Enable RAG-enhanced fuzzing with dataflow analysis (requires --api-key).", 
                      action="store_true", dest="use_rag")
    parser.add_argument("--audit-file", help="Path to audit report for context-aware LLM fuzzing.", 
                      action="store", dest="audit_file", type=str)
    parser.add_argument("--skip-dataflow-analysis", help="Skip dataflow analysis even when using RAG.", 
                      action="store_true", dest="skip_dataflow")

    parser.add_argument("--data-dependency",
                        help="Disable/Enable data dependency analysis: 0 - Disable, 1 - Enable (default: 1)",
                        action="store",
                        dest="data_dependency", type=int, default=1)
    parser.add_argument("--constraint-solving",
                        help="Disable/Enable constraint solving: 0 - Disable, 1 - Enable (default: 1)", action="store",
                        dest="constraint_solving", type=int)
    parser.add_argument("--environmental-instrumentation",
                        help="Disable/Enable environmental instrumentation: 0 - Disable, 1 - Enable (default: 1)",
                        action="store",
                        dest="environmental_instrumentation", type=int)
    parser.add_argument("--max-individual-length",
                        help="Maximal length of an individual (default: " + str(settings.MAX_INDIVIDUAL_LENGTH) + ")",
                        action="store",
                        dest="max_individual_length", type=int)
    parser.add_argument("--max-symbolic-execution",
                        help="Maximum number of symbolic execution calls before restting population (default: " + str(
                            settings.MAX_SYMBOLIC_EXECUTION) + ")", action="store",
                        dest="max_symbolic_execution", type=int)

    # cross contract fuzz parameters
    parser.add_argument("--cross-contract", type=int, help="open cross contract mode, open -- 1, close -- 2 (default)",
                        action="store", dest="cross_contract", default=2)
    parser.add_argument("--depend-contracts", type=str, nargs="*",
                        help="main fuzzed contract depend those contracts, you should give some names.",
                        dest="depend_contracts")
    parser.add_argument("--trans-json-path", type=str, help="location to save trans info to json",
                        dest="trans_json_path")
    parser.add_argument("--solc-path-cross", type=str, help="solc path, used by cross-slither", dest="solc_path_cross")
    parser.add_argument("--constructor-args", type=str, nargs="*",
                        help="constructor args, like: [address, uint, .....]", dest="constructor_args")
    parser.add_argument("--open-trans-comp", type=int, help="open cross trans mode, open -- 1 (default), close -- 2",
                        action="store", dest="trans_comp", default=1)
    parser.add_argument("--trans-mode", type=int, help="trans support mode, open other -- 1, no exec other -- 2",
                        default=1, dest="trans_mode")
    parser.add_argument("--p-open-cross", type=int, help="use cross trans probability: (1~8)", default=5,
                        dest="p_open_cross")
    parser.add_argument("--cross-init-mode", type=int, help="cross init mode: 1 -- specify, 2 -- random, 3 -- close",
                        default=1, dest="cross_init_mode")
    parser.add_argument("--duplication", type=str, help="duplication mode: 0 -- close, 1 -- open", default='0',
                        dest="duplication")

    version = "ConFuzzius - Version 0.0.2 - "
    version += "\"By three methods we may learn wisdom:\n"
    version += "First, by reflection, which is noblest;\n"
    version += "Second, by imitation, which is easiest;\n"
    version += "And third by experience, which is the bitterest.\"\n"
    parser.add_argument("-v", "--version", action="version", version=version)

    args = parser.parse_args()

    if not args.contract:
        args.contract = ""

    if args.source and args.contract.startswith("0x"):
        parser.error("--source requires --contract to be a name, not an address.")
    if args.source and args.blockchain_state and args.blockchain_state.isnumeric():
        parser.error("--source requires --blockchain-state to be a file, not a number.")

    if args.abi and not args.contract.startswith("0x"):
        parser.error("--abi requires --contract to be an address, not a name.")
    if args.abi and args.blockchain_state and not args.blockchain_state.isnumeric():
        parser.error("--abi requires --blockchain-state to be a number, not a file.")

    # Kiểm tra logic RAG vs LLM
    if args.use_rag and args.use_llm:
        parser.error("Cannot use both --use-rag and --use-llm together. Choose one method.")
    if (args.use_rag or args.use_llm) and not args.api_key:
        parser.error("Both RAG and LLM enhanced fuzzing require --api-key parameter.")

    if args.evm_version:
        settings.EVM_VERSION = args.evm_version
    if not args.solc_version:
        args.solc_version = solcx.get_solc_version()
    if args.generations:
        settings.GENERATIONS = args.generations
    if args.global_timeout:
        settings.GLOBAL_TIMEOUT = args.global_timeout
    if args.population_size:
        settings.POPULATION_SIZE = args.population_size
    if args.probability_crossover:
        settings.PROBABILITY_CROSSOVER = args.probability_crossover
    if args.probability_mutation:
        settings.PROBABILITY_MUTATION = args.probability_mutation

    if args.data_dependency is None:
        args.data_dependency = 1
    if args.constraint_solving is None:
        args.constraint_solving = 1
    if args.environmental_instrumentation is None:
        args.environmental_instrumentation = 1

    if args.environmental_instrumentation == 1:
        settings.ENVIRONMENTAL_INSTRUMENTATION = True
    elif args.environmental_instrumentation == 0:
        settings.ENVIRONMENTAL_INSTRUMENTATION = False

    if args.max_individual_length:
        settings.MAX_INDIVIDUAL_LENGTH = args.max_individual_length
    if args.max_symbolic_execution:
        settings.MAX_SYMBOLIC_EXECUTION = args.max_symbolic_execution

    if args.abi:
        settings.REMOTE_FUZZING = True

    if args.rpc_host:
        settings.RPC_HOST = args.rpc_host
    if args.rpc_port:
        settings.RPC_PORT = args.rpc_port

    # cross contract
    if args.contract is None or args.contract == "" or args.cross_contract == 2:
        args.cross_contract = 2  # close
        args.depend_contracts = []
        args.trans_json_path = None
    else:
        if args.contract is None or args.contract == "":
            print(
                '\033[42;31m!!!!!!if open cross contract mode, you need specify a main contract which will be fuzzed!!!!!!\033[0m')
            print('\033[42;31m!!!!!!use --contract [Example]!!!!!!\033[0m')
            sys.exit(-1)
        if args.depend_contracts is None:
            print(
                '\033[42;31m!!!!!!if open cross contract mode, you need specify some contract names which depended by main contract!!!!!!\033[0m')
            print('\033[42;31m!!!!!!use --depend-contracts [A B C]!!!!!!\033[0m')
            sys.exit(-1)
        if args.constructor_args is None:
            print('\033[42;31m!!!!!!if open cross contract mode, you need specify some constructor args!!!!!!\033[0m')
            print('\033[42;31m!!!!!!use --constructor-args [address, uint, .....]!!!!!!\033[0m')
            sys.exit(-1)
    if args.trans_json_path is not None:
        settings.TRANS_INFO_JSON_PATH = args.trans_json_path
        print(f'\033[42;31m!!!!!!设置用于存储事务序列信息的json地址{settings.TRANS_INFO_JSON_PATH}!!!!!!\033[0m')
        if os.path.exists(settings.TRANS_INFO_JSON_PATH):
            print(
                f'\033[42;31m!!!!!!用于存储事务序列信息的json地址{settings.TRANS_INFO_JSON_PATH}已经存在了, 现已覆盖!!!!!!\033[0m')
    if args.trans_comp == 1:
        settings.TRANS_COMP_OPEN = True  # 是否开启反馈机制
    elif args.trans_comp == 2:
        settings.TRANS_COMP_OPEN = False
    settings.MAIN_CONTRACT_NAME = args.contract
    settings.SOLC_PATH_CROSS = args.solc_path_cross
    settings.P_OPEN_CROSS = args.p_open_cross
    settings.CROSS_INIT_MODE = args.cross_init_mode
    settings.TRANS_SUPPORT_MODE = args.trans_mode
    if args.duplication == '0':
        settings.DUPLICATION = True
    else:
        settings.DUPLICATION = False
    if args.cross_contract == 1 and settings.SOLC_PATH_CROSS is None:
        print('\033[42;31m!!!!!!you need specify a solc path!!!!!!\033[0m')
        sys.exit(-1)

    return args


if '__main__' == __name__:
    main()

import subprocess
import os
import logging
import json
import threading
import requests
import time
import sys
from comp import analysis_depend_contract, analysis_main_contract_constructor
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_analysis(api_key, contract_path):
    """Step 1: Chạy Analysis.py để phân tích hợp đồng thông minh."""
    analysis_cmd = [
        "python", "./SmartSemanticAnalyzer/Analysis.py",
        "--server-url", "http://localhost:5000",
        "--contract-path", contract_path,
        "--solc-path", "/usr/bin/solc",
        "--output", "analysis_output.txt"
    ]
    result = subprocess.run(analysis_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Analysis failed: {result.stderr}")
        return False
    
    # Kiểm tra xem file có được tạo ra không
    if not os.path.exists("analysis_output.txt"):
        logging.error("Analysis output file was not created")
        return False
        
    logging.info("Analysis completed successfully.")
    return True  # Trả về True nếu file tồn tại

def run_rag_ask(api_key, question):
    """Gửi câu hỏi tới rag_googleapi.py để nhận phản hồi."""
    ask_cmd = [
        "python", "./RAG/rag_googleapi.py", "ask",
        "--api-key", api_key,
        "--question", question
    ]
    result = subprocess.run(ask_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"RAG ask failed: {result.stderr}")
        return None
    logging.info("RAG ask completed successfully.")
    return result.stdout

def generate_constructor_params(api_key, analysis_file="analysis_output.txt", save_path="constructor_params.txt"):
    if not os.path.exists(analysis_file):
        logging.error("Analysis output file does not exist.")
        return None

    try:
        with open(analysis_file, "r", encoding="utf-8") as f:
            analysis_output = f.read()
        print(analysis_output)
        logging.info("Successfully read analysis output from file.")
    except Exception as e:
        logging.error(f"Failed to read analysis output: {e}")
        return None

    prompt = (
        f"Below is the analysis result of a Solidity smart contract:\n\n"
        f"{analysis_output}\n\n"
        "Based on the constructor(s) of the main contract and this analysis, generate a valid constructor_params.json has type and random value for fuzzing smart contract base on your audit report"
        "file content in the following format, matching the structure and types of the constructor parameters:\n\n"
        "{\n"
        '  "_sub": {\n'
        '    "type": "contract",\n'
        '    "value": "Sub"\n'
        '  },\n'
        '  "_p": {\n'
        '    "type": "uint256",\n'
        '    "value": 12\n'
        '  }\n'
        "}\n\n"
        "Return ONLY this JSON object. Do not include any other text or explanation."
    )


    response = run_rag_ask(api_key, prompt)
    if not response:
        logging.error("Failed to get response from RAG for constructor params.")
        return None
    try:
        # json_start = response.find("{")
        # json_end = response.rfind("}") + 1
        # if json_start == -1 or json_end == -1:
        #     raise ValueError("JSON array not found in response.")
        # json_str = response[json_start:json_end]
        # constructor_params = json.loads(json_str)
        # logging.info("Successfully parsed constructor params JSON.")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=4)
        logging.info(f"Saved constructor parameters to {save_path}")
        return save_path

    except Exception as e:
        logging.error(f"Failed to parse RAG response for constructor params: {e}")
        return None

def generate_crossfuzz_input(api_key, output_file="analysis_output.txt", save_path="crossfuzz_input.json"):
    """Step 2: Đọc báo cáo kiểm toán và tạo đầu vào CrossFuzz bằng RAG."""    
    if not os.path.exists(output_file):
        logging.error("Analysis output file does not exist.")
        return None

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            analysis_output = f.read()
        logging.info("Successfully read analysis output from file.")
    except Exception as e:
        logging.error(f"Failed to read analysis output: {e}")
        return None

    prompt = (
        f"Below is the analysis for the smart contract written in Solidity:\n\n"
        f"{analysis_output}\n\n"
        "Based on this analysis and your security audit, generate optimized input arguments for running CrossFuzz.py.\n"

        "The required arguments are:\n"
        # "- sol file path (`p`): assume it's available and named in your output as `contract.sol`\n"
        "- contract name (`c_name`)\n"
        "- solc_version \n"
        "- max_trans_length (a reasonable integer > 0, not null)\n"
        "- fuzz_time (integer > 0, in seconds, not null)\n"
        # "- constructor_params_path (return `auto` if constructor has no complex parameters, or provide a JSON-style param path if needed)\n"
        "- trans_duplication (0 or 1, depending on whether transaction duplication is needed for fuzzing)\n\n"
        "Return ONLY the result in pure JSON format with exactly these fields:\n"
        "{\n"
        '  "c_name": "...",\n'
        '  "solc_version": "...",\n'
        '  "max_trans_length": ..., \n'
        '  "fuzz_time": ..., \n'
        # '  "constructor_params_path": "./constructor_params.json",\n'
        '  "trans_duplication": ...\n'
        "}\n"
        "Do not add explanations or comments. Return only valid JSON."
    )

    response = run_rag_ask(api_key, prompt)
    if not response:
        logging.error("Failed to get response from RAG.")
        return None
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("JSON object not found in response.")
        json_str = response[json_start:json_end]
        crossfuzz_input = json.loads(json_str)
        logging.info("Successfully parsed CrossFuzz input JSON.")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(crossfuzz_input, f, indent=4)
        logging.info(f"Saved CrossFuzz input to {save_path}")
        return save_path

    except Exception as e:
        logging.error(f"Failed to parse RAG response as JSON: {e}")
        return None

def run_crossfuzz_with_shell_script(input_json_path, contract_path, solc_path):
    import subprocess, logging

    cmd = ["bash", "run_crossfuzz.sh", input_json_path, contract_path, solc_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in iter(process.stdout.readline, ''):
        print(line.strip())

    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        logging.error(f"CrossFuzz shell script failed with return code {return_code}")
        return False

    logging.info("CrossFuzz executed successfully via shell script.")
    return True

def run_rag_enhanced_fuzzing(
    api_key: str,
    contract_path: str,
    solc_path: str,
    solc_version: str = "0.8.26",
    max_trans_length: int = 10,
    fuzz_time: int = 60,
    constructor_params_path: str = "auto",
    duplication: str = "0",
    audit_file: str = None,
    results_path: str = "fuzzing_results.json"
) -> bool:
    """
    Step 2: Run RAG + Dataflow enhanced fuzzing via fuzzer/main.py
    Args:
        api_key: Google API Key for RAG server
        contract_path: path to the .sol file
        solc_path: path to solc binary
        solc_version: compiler version (e.g. "0.8.26")
        max_trans_length: maximum transaction sequence length
        fuzz_time: fuzzing duration in seconds
        constructor_params_path: "auto" or path to JSON constructor args
        duplication: "0" or "1" to allow duplicate transactions
        audit_file: optional audit report for context
        results_path: path to save fuzzing results
    """
    logging.info("Step 2: Running RAG + Dataflow enhanced fuzzing...")

    # 1. Ensure RAG server is running
    def is_server_running():
        try:
            return requests.get("http://localhost:5000/health", timeout=2).status_code == 200
        except:
            return False

    if not is_server_running():
        logging.info("Starting RAG server...")
        def start_server():
            env = os.environ.copy()
            env["GOOGLE_API_KEY"] = api_key
            subprocess.Popen(
                ["python", "RAG/server.py"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        threading.Thread(target=start_server, daemon=True).start()
        for i in range(10):
            logging.info(f"Waiting for RAG server to start... ({i+1}/10)")
            if is_server_running():
                logging.info("RAG server is running!")
                break
            time.sleep(2)
        else:
            logging.warning("RAG server did not start in time; continuing anyway.")
    else:
        logging.info("RAG server is already running.")

    # 2. Get contract name from path
    contract_name = os.path.basename(contract_path).split(".")[0]
    
    # 3. Analyze dependent contracts
    logging.info("Analyzing dependent contracts...")
    depend_contracts, sl = analysis_depend_contract(
        file_path=contract_path,
        _contract_name=contract_name,
        _solc_version=solc_version,
        _solc_path=solc_path
    )
    
    # Quyết định chế độ cross contract dựa trên kết quả phân tích
    cross_contract_mode = "1"  # Mặc định bật
    if not depend_contracts or len(depend_contracts) <= 0:
        logging.warning("No dependent contracts found, disabling cross-contract mode")
        depend_contracts = []  # Đảm bảo depend_contracts là list rỗng
        cross_contract_mode = "2"  # Tắt chế độ cross-contract
    
    # Chuyển depend_contracts từ set sang list nếu cần
    if isinstance(depend_contracts, set):
        depend_contracts = list(depend_contracts)

    # 4. Process constructor arguments
    logging.info("Processing constructor arguments...")
    constructor_args = []  # Danh sách cuối cùng sẽ chứa các tham số theo format [name, type, value]
    
    if constructor_params_path != "auto":
        try:
            with open(constructor_params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            
            # Chuyển đổi từ JSON format sang format [name, type, value] mà deploy_contract yêu cầu
            for param_name, param_details in params.items():
                constructor_args.append(param_name)                  # name
                constructor_args.append(param_details["type"])       # type
                constructor_args.append(str(param_details["value"])) # value
                
        except Exception as e:
            logging.error(f"Failed to read constructor params: {e}")
            return False
    else:
        # analysis_main_contract_constructor trả về danh sách param dạng dictionary
        # cần chuyển thành dạng [name, type, value]
        raw_params = analysis_main_contract_constructor(
            file_path=contract_path,
            _contract_name=contract_name,
            sl=sl
        )
        
        if raw_params is None:
            logging.warning("No constructor parameters found")
            # Không cần return False, chúng ta có thể tiếp tục với constructor trống
            constructor_args = []
        else:
            # Xử lý trường hợp raw_params là chuỗi thay vì list dictionary
            if isinstance(raw_params, str):
                logging.warning(f"Constructor parameters returned as string: {raw_params}")
                constructor_args = []
            elif len(raw_params) > 0 and isinstance(raw_params[0], str):
                # Chuỗi có định dạng "name type value"
                logging.info("Converting string format parameters to [name, type, value] format")
                for param_str in raw_params:
                    parts = param_str.split(" ", 2)
                    if len(parts) == 3:
                        constructor_args.append(parts[0])  # name
                        constructor_args.append(parts[1])  # type
                        constructor_args.append(parts[2])  # value
                    else:
                        logging.warning(f"Malformed constructor parameter string: {param_str}")
            else:
                # Chuyển từ list dictionary sang format [name, type, value]
                for param in raw_params:
                    constructor_args.append(param["name"])                  # name
                    constructor_args.append(param["type"])                  # type
                    # Xử lý giá trị mặc định nếu value là None
                    param_value = "YA_DO_NOT_KNOW" if param["value"] is None else str(param["value"])
                    constructor_args.append(param_value)                    # value
    
    # Log analysis results
    logging.info("=== Analysis Results ===")
    logging.info(f"Dependent contracts: {depend_contracts}")
    logging.info(f"Constructor args (raw format): {constructor_args}")
    
    # Log constructor args theo từng tham số để dễ kiểm tra
    if constructor_args:
        logging.info("Constructor arguments (formatted):")
        for i in range(0, len(constructor_args), 3):
            if i+2 < len(constructor_args):
                logging.info(f"  Param: {constructor_args[i]}, Type: {constructor_args[i+1]}, Value: {constructor_args[i+2]}")
    
    logging.info(f"Cross-contract mode: {cross_contract_mode}")
    logging.info("======================")

    # 5. Prepare audit-file param
    audit_param = []
    if audit_file and os.path.exists(audit_file):
        audit_param = ["--audit-file", audit_file]
    
    # 6. Build and run fuzzer command with correct parameters based on main.py
    cmd = [
        "python", "fuzzer/main.py",
        "-s", contract_path,  # Sử dụng -s thay vì --source
        "-c", contract_name,  # Sử dụng -c thay vì --contract
        "--solc", f"v{solc_version}",
        "--solc-path-cross", solc_path,
        "-n", "36",  # Kích thước quần thể
        "-t", str(fuzz_time),  # Sử dụng -t thay vì --fuzz-time
        "--max-individual-length", str(max_trans_length),
        "--duplication", duplication,
        "--api-key", api_key,
        "--use-rag",
        "--cross-contract", cross_contract_mode  # Sử dụng cross_contract_mode (1=bật, 2=tắt)
    ]
    
    # Thêm constructor_args (nếu có)
    if constructor_args:
        cmd.append("--constructor-args")
        cmd.extend(constructor_args)
    
    # Thêm depend_contracts (chỉ khi có và cross_contract_mode = 1)
    if depend_contracts and cross_contract_mode == "1":
        cmd.append("--depend-contracts")
        cmd.extend(depend_contracts)
    
    # Thêm các tham số khác
    cmd.extend(audit_param)
    cmd.extend(["-r", results_path])  # Sử dụng -r thay vì --results

    # Log command line
    logging.info(f"Running command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Stream output in real-time
    for line in iter(proc.stdout.readline, ""):
        print(line.strip())
    proc.stdout.close()
    rc = proc.wait()

    if rc != 0:
        logging.error(f"RAG-enhanced fuzzing failed (exit code {rc})")
        return False

    logging.info("RAG-enhanced fuzzing completed successfully")
    return True

def run_llm_enhanced_fuzzing(api_key, contract_path, solc_path, audit_file=None):
    """Chạy fuzzing với LLM enhancement trực tiếp từ main.py"""
    logging.info("Running LLM-enhanced fuzzing...")
    
    # Đọc báo cáo audit nếu có
    audit_param = []
    if audit_file and os.path.exists(audit_file):
        audit_param = ["--audit-file", audit_file]
        
    # Lấy tên contract từ đường dẫn file
    contract_name = os.path.basename(contract_path).split('.')[0]
    
    # Thêm các hợp đồng phụ thuộc dựa trên tên hợp đồng
    depend_contracts = []
    if contract_name == "BECToken" or contract_name == "BecToken":
        depend_contracts = ["SafeMath", "ERC20Basic", "BasicToken", "ERC20", 
                           "StandardToken", "Ownable", "Pausable", "PausableToken"]
    
    # Cấu hình và chạy fuzzer
    cmd = [
        "python", "fuzzer/main.py",
        "--source", contract_path,
        "--contract", contract_name,  # Sử dụng tên contract đã lấy
        "--solc", "v0.8.26",  # Chỉ định phiên bản solc
        "--solc-path-cross", solc_path,  # Thêm đường dẫn solc cho cross contract fuzzing
        "--cross-contract", "1",  # Kích hoạt cross contract fuzzing
        "--depend-contracts"
    ] + depend_contracts + [  # Thêm danh sách hợp đồng phụ thuộc
        "--api-key", api_key,
        "--use-llm",
        "--constructor-args", "auto"  # Tự động phát hiện các tham số constructor
    ] + audit_param
    
    logging.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in iter(process.stdout.readline, ''):
        print(line.strip())
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        logging.error(f"LLM-enhanced fuzzing failed with return code {return_code}")
        return False
    
    logging.info("LLM-enhanced fuzzing completed successfully")
    return True

def run_integration(
    api_key: str,
    contract_path: str,
    solc_path: str,
    solc_version: str = "0.8.26",
    max_trans_length: int = 10,
    fuzz_time: int = 60,
    constructor_params: str = "auto",
    duplication: str = "0",
    audit_file: str = None,
    results_path: str = "fuzzing_results.json",
    use_crossfuzz: bool = False,
    use_rag: bool = False
):
    logging.info("Step 1: Running Analysis.py...")
    if not run_analysis(api_key, contract_path):
        logging.error("Aborting: analysis failed.")
        return
        
    # Use analysis output as audit report if none provided
    if audit_file is None and os.path.exists("analysis_output.txt"):
        audit_file = "analysis_output.txt"
        logging.info("Using analysis output as audit report for LLM context")
    
    if use_crossfuzz:
        # Legacy CrossFuzz method
        logging.info("Step 2: Generating constructor params with RAG...")
        constructor_params_path = generate_constructor_params(api_key)
        if not constructor_params_path:
            logging.error("Failed to generate constructor params.")
            return
            
        logging.info("Step 3: Generating CrossFuzz input...")
        crossfuzz_input_path = generate_crossfuzz_input(api_key)
        if not crossfuzz_input_path:
            logging.error("Failed to generate CrossFuzz input.")
            return
            
        logging.info("Step 4: Running CrossFuzz...")
        if not run_crossfuzz_with_shell_script(crossfuzz_input_path, contract_path, solc_path):
            logging.error("CrossFuzz failed.")
            return
    elif use_rag:
        # New RAG + Dataflow method
        logging.info("Step 2: Running RAG + Dataflow enhanced fuzzing...")
        success = run_rag_enhanced_fuzzing(
            api_key=api_key,
            contract_path=contract_path,
            solc_path=solc_path,
            solc_version=solc_version,
            max_trans_length=max_trans_length,
            fuzz_time=fuzz_time,
            constructor_params_path=constructor_params,
            duplication=duplication,
            audit_file=audit_file,
            results_path=results_path
        )
        if not success:
            logging.error("RAG-enhanced fuzzing failed.")
            return
    else:
        # LLM method
        logging.info("Step 2: Running LLM-enhanced fuzzing...")
        if not run_llm_enhanced_fuzzing(api_key, contract_path, solc_path, audit_file):
            logging.error("LLM-enhanced fuzzing failed.")
            return

    logging.info("=== Integration completed successfully ===")

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Integrate Analysis + RAG/LLM fuzzing for Solidity contracts"
    )
    parser.add_argument(
        "--api-key", required=True,
        help="Google/OpenAI API Key for RAG server"
    )
    parser.add_argument(
        "--contract-path", "-s", required=True,
        help="Path to the Solidity source file"
    )
    parser.add_argument(
        "--solc-path", required=True,
        help="Path to solc binary"
    )
    parser.add_argument(
        "--solc-version", default="0.8.26",
        help="Solidity compiler version (e.g. 0.4.26, 0.8.26)"
    )
    parser.add_argument(
        "--max-trans-length", "-m", type=int, default=10,
        help="Maximum transaction sequence length"
    )
    parser.add_argument(
        "--fuzz-time", "-t", type=int, default=60,
        help="Total fuzzing time in seconds"
    )
    parser.add_argument(
        "--constructor-params", "-cargs", default="auto",
        help="'auto' or path to JSON file of constructor parameters"
    )
    parser.add_argument(
        "--duplication", "-d", default="0",
        choices=["0", "1"],
        help="Allow duplicate transactions? 0=no, 1=yes"
    )
    parser.add_argument(
        "--audit-file", "-a",
        help="Optional audit report file to seed RAG context"
    )
    parser.add_argument(
        "--results", default="fuzzing_results.json",
        help="Path to save fuzzing results"
    )
    parser.add_argument(
        "--use-crossfuzz", action="store_true",
        help="Run legacy CrossFuzz workflow instead of RAG"
    )
    parser.add_argument(
        "--use-rag", action="store_true",
        help="Run RAG + Dataflow enhanced fuzzing"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.contract_path):
        logging.error(f"Contract file not found: {args.contract_path}")
        sys.exit(1)
    if not os.path.isfile(args.solc_path):
        logging.error(f"Solc binary not found: {args.solc_path}")
        sys.exit(1)

    # Run integration
    run_integration(
        api_key=args.api_key,
        contract_path=args.contract_path,
        solc_path=args.solc_path,
        solc_version=args.solc_version,
        max_trans_length=args.max_trans_length,
        fuzz_time=args.fuzz_time,
        constructor_params=args.constructor_params,
        duplication=args.duplication,
        audit_file=args.audit_file,
        results_path=args.results,
        use_crossfuzz=args.use_crossfuzz,
        use_rag=args.use_rag
    )
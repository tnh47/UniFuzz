import subprocess
import os
import logging
import json
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_analysis(api_key, contract_path):
    """Step 1: Chạy Analysis.py để phân tích hợp đồng thông minh."""
    analysis_cmd = [
        "python", "./SmartSemanticAnalyzer/Analysis.py",
        "--api-key", api_key,
        "--contract-path", contract_path,
        "--output", "analysis_output.txt"
    ]
    result = subprocess.run(analysis_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Analysis failed: {result.stderr}")
        return False
    logging.info("Analysis completed successfully.")
    return os.path.exists("analysis_output.txt")

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

def run_rag_enhanced_fuzzing(api_key, contract_path, solc_path, audit_file=None):
    """Chạy fuzzing với RAG + Dataflow enhancement từ main.py"""
    logging.info("Running RAG + Dataflow enhanced fuzzing...")
    
    # Kiểm tra và khởi động RAG server nếu chưa chạy
    try:
        import requests
        import subprocess
        import time
        import threading
        import os
        
        def is_server_running():
            try:
                response = requests.get("http://localhost:5000/health", timeout=2)
                return response.status_code == 200
            except:
                return False
        
        if not is_server_running():
            logging.info("Starting RAG server...")
            
            def start_server():
                env = os.environ.copy()
                env["GOOGLE_API_KEY"] = api_key
                try:
                    subprocess.Popen(["python", "RAG/server.py"], 
                                    env=env, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
                except Exception as e:
                    logging.error(f"Failed to start RAG server: {e}")
            
            server_thread = threading.Thread(target=start_server)
            server_thread.daemon = True
            server_thread.start()
            
            # Đợi server khởi động
            max_retries = 10
            for i in range(max_retries):
                logging.info(f"Waiting for RAG server to start... ({i+1}/{max_retries})")
                if is_server_running():
                    logging.info("RAG server is running!")
                    break
                time.sleep(2)
            else:
                logging.warning("Could not verify if RAG server is running. Will continue anyway.")
        else:
            logging.info("RAG server is already running.")
    except Exception as e:
        logging.error(f"Error while checking/starting RAG server: {e}")
        logging.warning("Will continue without verifying RAG server status.")
    
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
    
    # Cấu hình và chạy fuzzer với RAG + Dataflow
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
        "--use-rag",  # Sử dụng RAG thay vì LLM
        "--constructor-args", "auto"  # Tự động phát hiện các tham số constructor
    ] + audit_param
    
    logging.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in iter(process.stdout.readline, ''):
        print(line.strip())
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        logging.error(f"RAG-enhanced fuzzing failed with return code {return_code}")
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

def run_integration(api_key, contract_path, solc_path, audit_file=None, use_crossfuzz=False, use_rag=False):
    logging.info("Step 1: Running Analysis.py...")
    if not run_analysis(api_key, contract_path):
        logging.error("Aborting: analysis failed.")
        return
        
    # Nếu bước phân tích thành công, có thể sử dụng phân tích làm báo cáo audit
    if audit_file is None and os.path.exists("analysis_output.txt"):
        audit_file = "analysis_output.txt"
        logging.info("Using analysis output as audit report for LLM context")
    
    if use_crossfuzz:
        # Phương pháp cũ: sử dụng CrossFuzz
        logging.info("Step 2: Generating constructor params with RAG...")
        constructor_params_path = generate_constructor_params(api_key)
        if not constructor_params_path:
            logging.error("Failed to generate constructor params.")
            
        logging.info("Step 3: Generating CrossFuzz input...")
        crossfuzz_input_path = generate_crossfuzz_input(api_key)
        if not crossfuzz_input_path:
            logging.error("Failed to generate CrossFuzz input.")
            
        logging.info("Step 4: Running CrossFuzz...")
        if not run_crossfuzz_with_shell_script(crossfuzz_input_path, contract_path, solc_path):
            logging.error("CrossFuzz failed.")
            return
    elif use_rag:
        # Phương pháp mới với RAG + Dataflow
        logging.info("Step 2: Running RAG + Dataflow enhanced fuzzing...")
        if not run_rag_enhanced_fuzzing(api_key, contract_path, solc_path, audit_file):
            logging.error("RAG-enhanced fuzzing failed.")
            return
    else:
        # Phương pháp với LLM
        logging.info("Step 2: Running LLM-enhanced fuzzing...")
        if not run_llm_enhanced_fuzzing(api_key, contract_path, solc_path, audit_file):
            logging.error("LLM-enhanced fuzzing failed.")
            return

    logging.info("=== Integration completed successfully ===")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG + Fuzzing Integration")
    parser.add_argument("--api-key", required=True, help="Google API Key")
    parser.add_argument("--contract-path", required=True, help="Path to Solidity contract")
    parser.add_argument("--solc-path", required=True, help="Path to solc binary")
    parser.add_argument("--audit-file", help="Path to audit report for additional context")
    parser.add_argument("--use-crossfuzz", action="store_true", help="Use CrossFuzz method instead of direct fuzzing")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG + Dataflow enhanced fuzzing (default is LLM-enhanced)")

    args = parser.parse_args()

    if not os.path.exists(args.contract_path):
        logging.error(f"Contract file not found: {args.contract_path}")
    elif not os.path.exists(args.solc_path):
        logging.error(f"solc not found: {args.solc_path}")
    else:
        run_integration(args.api_key, args.contract_path, args.solc_path, args.audit_file, args.use_crossfuzz, args.use_rag)

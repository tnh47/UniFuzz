import subprocess
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_analysis(api_key, contract_path):
    """Chạy Analysis.py để phân tích hợp đồng thông minh."""
    analysis_cmd = [
        "python", "./SmartSemanticAnalyzer/Analysis.py",
        "--api-key", api_key,
        "--contract-path", contract_path
    ]
    result = subprocess.run(analysis_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Analysis failed: {result.stderr}")
        return False
    logging.info("Analysis completed successfully.")
    return os.path.exists("analysis_output.txt")

def run_rag_ask(api_key, question):
    """Chạy rag_googleapi.py ask để tạo seed fuzzing."""
    ask_cmd = [
        "python", "./RAG/rag_googleapi.py", "ask",
        "--api-key", api_key,
        "--question", question
    ]
    result = subprocess.run(ask_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"RAG ask failed: {result.stderr}")
        return False
    logging.info("RAG ask completed successfully.")
    print("RAG Response:")
    print(result.stdout)
    return True

def run_integration(api_key, contract_path):
    """Tích hợp toàn bộ quy trình."""
    # Bước 1: Chạy Analysis.py
    logging.info("Step 1: Running smart contract analysis...")
    if not run_analysis(api_key, contract_path):
        logging.error("Aborting due to analysis failure.")
        return

    # Đọc output từ bước 1
    output_file = "analysis_output.txt"
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                analysis_output = f.read()
            logging.info("Successfully read analysis output from file.")
        except Exception as e:
            logging.error(f"Failed to read analysis output: {e}")
            return
    else:
        logging.error(f"Analysis output file not found: {output_file}")
        return

    # Bước 2: Hỏi RAG với câu hỏi tích hợp output từ bước 1
    question_base = "Based on the following analysis, generate fuzzing seeds for the identified vulnerable functions:\n\n"
    question = question_base + analysis_output
    logging.info("Step 2: Querying RAG for fuzzing seeds with analysis output...")
    if not run_rag_ask(api_key, question):
        logging.error("Aborting due to ask failure.")
        return

    logging.info("Integration completed successfully.")

if __name__ == "__main__":
    import argparse

    # Thiết lập đối số dòng lệnh
    parser = argparse.ArgumentParser(description="Integrate smart contract analysis and RAG for fuzzing seeds")
    parser.add_argument("--api-key", required=True, help="Google API Key")
    parser.add_argument("--contract-path", required=True, help="Path to the smart contract file")

    args = parser.parse_args()

    # Lấy API key và contract path từ đối số
    api_key = args.api_key
    contract_path = args.contract_path

    # Kiểm tra file hợp đồng có tồn tại không
    if not os.path.exists(contract_path):
        logging.error(f"Smart contract file not found: {contract_path}")
    else:
        run_integration(api_key, contract_path)
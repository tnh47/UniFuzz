import json
import subprocess
import logging
import os
import argparse
from google import genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_slither_data(contract_path):
    """Run Slither to extract AST, CFG, and DFG"""
    result = subprocess.run(
        ["slither", contract_path, "--json", "slither_output.json"],
        capture_output=True
    )
    if result.returncode != 0:
        logging.error(f"Slither failed with error: {result.stderr.decode()}")
    else:
        logging.info("Slither analysis completed successfully.")

def analyze_with_gemini(prompt, api_key):
    """Send a prompt to Gemini API for analysis"""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt
        )
        return response.text
    except Exception as e:
        logging.error("Error querying Gemini API: %s", e)
        return None

def main():
    # Thiết lập argparse để nhận đối số dòng lệnh
    parser = argparse.ArgumentParser(description="Smart Contract Analysis with Slither and Gemini")
    parser.add_argument("--api-key", help="Google API Key (optional, can use GOOGLE_API_KEY env var)")
    parser.add_argument("--contract-path", required=True, help="Path to the smart contract file")
    
    args = parser.parse_args()

    # Lấy API key từ đối số hoặc biến môi trường
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logging.error("API Key not found. Provide it via --api-key or set GOOGLE_API_KEY env variable.")
        return

    # Lấy đường dẫn smart contract từ đối số
    contract_path = args.contract_path
    if not os.path.exists(contract_path):
        logging.error(f"Smart contract file not found at: {contract_path}")
        return

    # Chạy Slither để phân tích hợp đồng
    extract_slither_data(contract_path)

    # Kiểm tra và đọc output từ Slither
    if os.path.exists("slither_output.json"):
        with open("slither_output.json", "r") as file:
            try:
                slither_data = json.load(file)
            except json.JSONDecodeError:
                logging.warning("Failed to parse Slither JSON output, continuing without it.")
                slither_data = {}
    else:
        logging.warning("Slither output file not found, continuing without it.")
        slither_data = {}

    # Nếu có dữ liệu từ Slither, phân tích bằng Gemini
    if slither_data:
        prompt = f"""
        Below is the raw security analysis output from Slither:
        {json.dumps(slither_data, indent=4)}
        
        Based on this data, please analyze the following:
        1. Which functions are most likely to contain critical security vulnerabilities?
        2. What key areas should be targeted for fuzz testing?
        3. What security recommendations can be provided to improve the contract?
        """
        
        gemini_response = analyze_with_gemini(prompt, api_key)
        if gemini_response:
            print("Analysis:")
            print(gemini_response)

            # Lưu kết quả vào file
            with open("analysis_output.txt", "w") as f:
                f.write("Analysis:\n")
                f.write(gemini_response)
            logging.info("Analysis output saved to analysis_output.txt")
        else:
            logging.error("Failed to get response from Gemini API.")

if __name__ == "__main__":
    main()
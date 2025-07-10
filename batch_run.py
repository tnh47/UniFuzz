import os
import subprocess
import re

# Cấu hình đầu vào
folder_path = "/home/ngonhat/Desktop/Source_code_SoliAudit"
solc_path = "/home/ngonhat/Desktop/UniFuzz/venv3.8/bin/solc"
solc_version = "0.4.26"
api_key = "AIzaSyAMtaEim5BMwMXPNanDTneelYUbxTXeiMI"
output_file = "summary_output-SoliAudit.txt"
run_count = 3

# Biểu thức chính quy để lọc log
pattern = re.compile(
    r"^INFO:Detector:(-+|.+!+\s+\?+\s+!+|-+|SWC-ID:.+|Severity:.+|Transaction sequence:|^$)"
    r"|^INFO:Analysis:(.+)$"
)

# Xóa file output cũ nếu có
if os.path.exists(output_file):
    os.remove(output_file)

# Duyệt tất cả các file .sol trong các folder con
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".sol"):
            contract_path = os.path.join(root, filename)
            relative_path = os.path.relpath(contract_path, folder_path)
            print(f"Đang chạy: {relative_path}")
            for i in range(run_count):
                print(f"  Lần chạy thứ {i+1}/{run_count}")
                try:
                    # Tạo lệnh chạy
                    command = [
                        "python", "run.py",
                        "--api-key", api_key,
                        "--contract-path", contract_path,
                        "--solc-path", solc_path,
                        "--solc-version", solc_version,
                        "--max-trans-length", "10",
                        "--fuzz-time", "160",
                        "--constructor-params", "auto",
                        "--duplication", "0",
                        "--use-rag",
                        "--results", "results.json"
                    ]

                    # Thực thi và thu thập output
                    result = subprocess.run(command, capture_output=True, text=True)
                    output_lines = result.stdout.splitlines()

                    # Ghi header và các dòng phù hợp
                    with open(output_file, "a") as f:
                        f.write(f"\n\n========== CONTRACT: {relative_path} - RUN: {i+1} ==========\n")
                        for line in output_lines:
                            if line.startswith("INFO:Detector:") or line.startswith("INFO:Analysis:"):
                                f.write(line + "\n")

                except Exception as e:
                    with open(output_file, "a") as f:
                        f.write(f"\n\n[ERROR] Lỗi khi chạy contract {relative_path}, lần {i+1}: {e}\n")

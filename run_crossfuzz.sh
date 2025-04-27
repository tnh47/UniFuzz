#!/bin/bash
# Usage: ./run_crossfuzz.sh crossfuzz_input.json contract.sol /path/to/solc

INPUT_JSON="$1"
CONTRACT_PATH="$2"
SOLC_PATH="$3"

# Kích hoạt Python 3.8 virtual environment
source venv3.8/bin/activate

# Trích xuất các tham số từ file JSON
CONTRACT_NAME=$(jq -r '.c_name' "$INPUT_JSON")
SOLC_VERSION=$(jq -r '.solc_version' "$INPUT_JSON")
MAX_TRANS_LEN=$(jq -r '.max_trans_length' "$INPUT_JSON")
FUZZ_TIME=$(jq -r '.fuzz_time' "$INPUT_JSON")
# CONSTRUCTOR_PARAMS_PATH=$(jq -r '.constructor_params_path' "$INPUT_JSON")
TRANS_DUP=$(jq -r '.trans_duplication' "$INPUT_JSON")

# Sử dụng solc-0.4.26 cho phiên bản Solidity 0.4.26
if [ "$SOLC_VERSION" = "0.4.26" ]; then
    SOLC_PATH="/usr/bin/solc-0.4.26"
else
    SOLC_PATH="/usr/bin/solc"
fi

if [[ "$CONSTRUCTOR_PARAMS_PATH" != "auto" && (! -s "$CONSTRUCTOR_PARAMS_PATH" || ! -f "$CONSTRUCTOR_PARAMS_PATH") ]]; then
    echo "constructor_params.json is empty or not found. Using 'auto'."
    CONSTRUCTOR_PARAMS_PATH="auto"
fi

# In ra terminal để debug
echo "== Running CrossFuzz with the following parameters =="
echo "SOL_FILE: $SOL_FILE"
echo "CONTRACT_NAME: $CONTRACT_NAME"
echo "SOLC_VERSION: $SOLC_VERSION"
echo "MAX_TRANS_LEN: $MAX_TRANS_LEN"
echo "FUZZ_TIME: $FUZZ_TIME"
echo "RES_PATH: ./result.json"
echo "SOLC_PATH: $SOLC_PATH"
echo "CONSTRUCTOR_PARAMS_PATH: $CONSTRUCTOR_PARAMS_PATH"
echo "TRANS_DUP: $TRANS_DUP"
echo "====================================================="

# Thực thi CrossFuzz.py
python CrossFuzz.py "$CONTRACT_PATH" "$CONTRACT_NAME" "$SOLC_VERSION" "$MAX_TRANS_LEN" "$FUZZ_TIME" \
    ./result.json "$SOLC_PATH" "$CONSTRUCTOR_PARAMS_PATH" "$TRANS_DUP"

#!/bin/bash

# Hàm cài đặt và chuyển phiên bản solc nếu cần
setup_solc() {
    local version=$1
    echo "Kiểm tra phiên bản solc $version..."
    
    # Kiểm tra xem phiên bản đã được cài đặt chưa
    if ! solc-select versions | grep -q "$version"; then
        echo "Cài đặt solc phiên bản $version..."
        solc-select install $version
        if [ $? -ne 0 ]; then
            echo "Không thể cài đặt solc phiên bản $version"
            return 1
        fi
    fi
    
    # Chuyển sang phiên bản đã chọn
    echo "Chuyển sang phiên bản solc $version..."
    solc-select use $version
    if [ $? -ne 0 ]; then
        echo "Không thể chuyển sang phiên bản solc $version"
        return 1
    fi
    
    echo "Đang sử dụng solc phiên bản: $(solc --version)"
    return 0
}

# Đường dẫn solc
SOLC_PATH=$(which solc)
echo "Đường dẫn solc: $SOLC_PATH"

# Lấy API key từ tham số dòng lệnh hoặc sử dụng giá trị mặc định
API_KEY=${1:-"YOUR_API_KEY"}

# Xác định phiên bản solc dựa trên tên hợp đồng
CONTRACT_PATH="./examples/BECToken.sol"
CONTRACT_NAME="BecToken"
SOLC_VERSION="0.4.16"

# Cài đặt phiên bản solc cần thiết
setup_solc $SOLC_VERSION
if [ $? -ne 0 ]; then
    echo "Không thể cài đặt hoặc chuyển sang phiên bản solc cần thiết. Thoát."
    exit 1
fi

# Chạy fuzzing với đầy đủ tham số
echo "Chạy fuzzing cho $CONTRACT_PATH với phiên bản solc $SOLC_VERSION..."
python fuzzer/main.py \
    --source $CONTRACT_PATH \
    --contract $CONTRACT_NAME \
    --solc "v$SOLC_VERSION" \
    --solc-path-cross $SOLC_PATH \
    --cross-contract 1 \
    --depend-contracts SafeMath ERC20Basic BasicToken ERC20 StandardToken Ownable Pausable PausableToken \
    --api-key "$API_KEY" \
    --use-llm \
    --constructor-args auto

echo "Fuzzing hoàn thành."
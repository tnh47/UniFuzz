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

# Hàm xác định phiên bản solc từ file solidity
get_solc_version() {
    local file_path=$1
    local version=$(grep -o "pragma solidity .*;" "$file_path" | sed -E 's/pragma solidity \^?([0-9]+\.[0-9]+\.[0-9]+).*/\1/')
    
    if [ -z "$version" ]; then
        version=$(grep -o "pragma solidity .*;" "$file_path" | sed -E 's/pragma solidity \^?([0-9]+\.[0-9]+).*/\1.0/')
    fi
    
    if [ -z "$version" ]; then
        echo "0.4.26"  # Phiên bản mặc định nếu không thể xác định
    else
        echo "$version"
    fi
}

# Hàm xác định các hợp đồng phụ thuộc
get_dependent_contracts() {
    local file_path=$1
    local contracts=$(grep -o "contract [A-Za-z0-9]* " "$file_path" | sed 's/contract //')
    echo "$contracts"
}

# Hiển thị cách sử dụng
usage() {
    echo "Cách sử dụng: $0 -c CONTRACT_PATH [-a API_KEY] [-n CONTRACT_NAME] [-v SOLC_VERSION]"
    echo "  -c CONTRACT_PATH   Đường dẫn đến file hợp đồng"
    echo "  -a API_KEY         Google API Key (tùy chọn)"
    echo "  -n CONTRACT_NAME   Tên hợp đồng chính (tùy chọn)"
    echo "  -v SOLC_VERSION    Phiên bản solc (tùy chọn)"
    echo "  -h                 Hiển thị trợ giúp này"
    exit 1
}

# Xử lý tham số dòng lệnh
CONTRACT_PATH=""
API_KEY=""
CONTRACT_NAME=""
SOLC_VERSION=""

while getopts "c:a:n:v:h" opt; do
    case ${opt} in
        c)
            CONTRACT_PATH=$OPTARG
            ;;
        a)
            API_KEY=$OPTARG
            ;;
        n)
            CONTRACT_NAME=$OPTARG
            ;;
        v)
            SOLC_VERSION=$OPTARG
            ;;
        h)
            usage
            ;;
        \?)
            echo "Tùy chọn không hợp lệ: -$OPTARG" 1>&2
            usage
            ;;
    esac
done

# Kiểm tra tham số bắt buộc
if [ -z "$CONTRACT_PATH" ]; then
    echo "Thiếu đường dẫn đến file hợp đồng!"
    usage
fi

# Nếu API_KEY không được cung cấp, sử dụng biến môi trường hoặc yêu cầu nhập
if [ -z "$API_KEY" ]; then
    if [ -n "$GOOGLE_API_KEY" ]; then
        API_KEY=$GOOGLE_API_KEY
    else
        echo "Nhập Google API Key:"
        read -r API_KEY
    fi
fi

# Lấy tên hợp đồng nếu không được chỉ định
if [ -z "$CONTRACT_NAME" ]; then
    CONTRACT_NAME=$(basename "$CONTRACT_PATH" .sol)
fi

# Lấy phiên bản solc nếu không được chỉ định
if [ -z "$SOLC_VERSION" ]; then
    SOLC_VERSION=$(get_solc_version "$CONTRACT_PATH")
    echo "Đã xác định phiên bản solc: $SOLC_VERSION"
fi

# Đường dẫn solc
SOLC_PATH=$(which solc)
echo "Đường dẫn solc: $SOLC_PATH"

# Cài đặt phiên bản solc cần thiết
setup_solc $SOLC_VERSION
if [ $? -ne 0 ]; then
    echo "Không thể cài đặt hoặc chuyển sang phiên bản solc cần thiết. Thoát."
    exit 1
fi

# Xác định các hợp đồng phụ thuộc
DEPEND_CONTRACTS=""
case "$CONTRACT_NAME" in
    "BECToken"|"BecToken")
        DEPEND_CONTRACTS="SafeMath ERC20Basic BasicToken ERC20 StandardToken Ownable Pausable PausableToken"
        ;;
    *)
        DEPEND_CONTRACTS=$(get_dependent_contracts "$CONTRACT_PATH")
        ;;
esac

# Nếu không có hợp đồng phụ thuộc, chỉ chạy không có tham số --depend-contracts
if [ -z "$DEPEND_CONTRACTS" ]; then
    DEPEND_ARGS=""
else
    DEPEND_ARGS="--depend-contracts $DEPEND_CONTRACTS"
fi

# Chạy fuzzing với đầy đủ tham số
echo "Chạy fuzzing cho $CONTRACT_PATH ($CONTRACT_NAME) với phiên bản solc $SOLC_VERSION..."
CMD="python fuzzer/main.py \
    --source $CONTRACT_PATH \
    --contract $CONTRACT_NAME \
    --solc v$SOLC_VERSION \
    --solc-path-cross $SOLC_PATH \
    --cross-contract 1 \
    $DEPEND_ARGS \
    --api-key \"$API_KEY\" \
    --use-llm \
    --constructor-args auto"

echo "Thực thi lệnh: $CMD"
eval $CMD

echo "Fuzzing hoàn thành."
# UniFuzz - Fuzzing nâng cao với RAG cho hợp đồng thông minh

UniFuzz là một công cụ fuzzing hợp đồng thông minh tích hợp kỹ thuật RAG (Retrieval Augmented Generation) để tăng hiệu quả phát hiện lỗi và tăng độ phủ mã.

## Kiến trúc hệ thống

Hệ thống UniFuzz bao gồm hai thành phần chính chạy song song trên hai phiên bản Python:

1. **CrossFuzz (Python 3.8)**: Công cụ fuzzing chính, đảm nhiệm việc phân tích và thực thi hợp đồng.
2. **RAG Server (Python 3.12+)**: Server Flask phục vụ yêu cầu RAG từ fuzzer, sử dụng Google Generative AI để tạo các test case thông minh.

Phương pháp này cho phép kết hợp ưu điểm của cả hai thành phần mà không gặp vấn đề về sự tương thích của phiên bản Python.

## Cài đặt

### 1. Thiết lập môi trường CrossFuzz (Python 3.8)

```bash
# Tạo môi trường ảo Python 3.8
python3.8 -m venv venv3.8
source venv3.8/bin/activate  # Linux/Mac
# hoặc
.\venv3.8\Scripts\activate  # Windows

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

### 2. Thiết lập môi trường RAG Server (Python 3.12+)

```bash
# Tạo môi trường ảo Python 3.12
python3.12 -m venv venv3.12
source venv3.12/bin/activate  # Linux/Mac
# hoặc
.\venv3.12\Scripts\activate  # Windows

# Cài đặt các gói phụ thuộc cho RAG
pip install -r RAG/requirements.txt
```

## Sử dụng

### 1. Khởi động RAG Server

```bash
# Trong terminal đầu tiên (với môi trường Python 3.12)
source venv3.12/bin/activate  # Linux/Mac
# hoặc
.\venv3.12\Scripts\activate  # Windows

# Cài đặt API key Google (nếu cần)
export GOOGLE_API_KEY="your-google-api-key"  # Linux/Mac
# hoặc
set GOOGLE_API_KEY="your-google-api-key"  # Windows

# Khởi động RAG server
python RAG/server.py
```

### 2. Chạy CrossFuzz với RAG

```bash
# Trong terminal thứ hai (với môi trường Python 3.8)
source venv3.8/bin/activate  # Linux/Mac
# hoặc
.\venv3.8\Scripts\activate  # Windows

# Chạy CrossFuzz với kết nối đến RAG server (đã chạy ở bước trước)
python fuzzer/main.py --source ./đường/dẫn/đến/hợp_đồng.sol --contract TênHợpĐồng --use-llm --api-key "your-google-api-key"
```

## Các tùy chọn nâng cao

### Sử dụng báo cáo audit để tăng hiệu quả RAG

```bash
python fuzzer/main.py --source ./đường/dẫn/đến/hợp_đồng.sol --contract TênHợpĐồng --use-llm --api-key "your-google-api-key" --audit-file ./path/to/audit_report.txt
```

### Lưu báo cáo hiệu quả RAG để phân tích

```bash
python fuzzer/main.py --source ./đường/dẫn/đến/hợp_đồng.sol --contract TênHợpĐồng --use-llm --api-key "your-google-api-key" --rag-effectiveness-file ./rag_effectiveness.json
```

## Hiệu quả của RAG so với fuzzing thông thường

Khi sử dụng RAG trong quá trình fuzzing, hệ thống có một số ưu điểm:

1. **Tăng hiệu quả phát hiện lỗi**: RAG giúp tạo ra các test case thông minh dựa trên kiểu dữ liệu và ngữ cảnh, tăng khả năng phát hiện lỗi so với sinh ngẫu nhiên.

2. **Tập trung vào các điểm yếu đã biết**: Bằng cách sử dụng báo cáo audit làm ngữ cảnh, RAG có thể tạo ra các giá trị đặc biệt nhắm vào các lỗ hổng tiềm ẩn.

3. **Cải thiện độ phủ mã**: Các test case thông minh giúp khám phá nhiều nhánh hơn trong mã nguồn.

4. **Thống kê hiệu quả**: Công cụ cung cấp báo cáo chi tiết về hiệu quả của RAG trong việc tăng coverage.

## Hoạt động của hệ thống

1. CrossFuzz gửi thông tin về hàm và tham số cần fuzzing đến RAG server thông qua HTTP.
2. RAG server sử dụng Google Generative AI để tạo giá trị tối ưu cho tham số.
3. CrossFuzz sử dụng giá trị này trong quá trình fuzzing và theo dõi tác động đến coverage.
4. Hệ thống tạo báo cáo về hiệu quả của RAG sau khi kết thúc fuzzing.

## Xử lý mất kết nối

Hệ thống có cơ chế xử lý mất kết nối giữa CrossFuzz và RAG server:

1. **Cache kết quả**: Các giá trị RAG được cache để tránh gọi lại nhiều lần.
2. **Fallback tự động**: Nếu không thể kết nối với RAG server, hệ thống tự động sử dụng giá trị ngẫu nhiên.
3. **Retry cơ bản**: Server không khả dụng sẽ được thử lại tối đa 3 lần trước khi fallback.

## Góp ý và báo lỗi

Nếu bạn gặp vấn đề hoặc có đề xuất cải tiến, vui lòng tạo issue trên kho lưu trữ của dự án.

### file generator.py la cua CrossFuzz
### 2 file llm_agent va llm_enhanced_generator la 2 file them vao 
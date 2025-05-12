from flask import Flask, request, jsonify
import os
import subprocess
import logging
import json
import hashlib
from functools import lru_cache
import threading
import time
import csv
from datetime import datetime
import re

app = Flask(__name__)

# Cấu hình logging chi tiết hơn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler("rag_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAGServer")

# Cấu hình
RAG_SCRIPT_PATH = "RAG/rag_googleapi.py"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyB3P2COlotMu-3RR-ehwZXZk60wOWJvfEA")  # Default empty string
REQUEST_TIMEOUT = 120  # seconds
CACHE_SIZE = 100  # Số lượng kết quả lớn nhất lưu trong cache
PERFORMANCE_LOG_FILE = "rag_performance.csv"  # File CSV để lưu thông tin hiệu suất
VULNERABILITY_LOG_FILE = "rag_vulnerabilities.csv"

# Cache kết quả để tránh gọi lại RAG nhiều lần cho cùng prompt
cache = {}
cache_lock = threading.Lock()

# Biến đếm cho thống kê
stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "timeouts": 0,
    "errors": 0,
    "success": 0,
    "start_time": time.time()
}
stats_lock = threading.Lock()

# Tạo file CSV nếu chưa tồn tại
def init_performance_log():
    """Khởi tạo file log hiệu suất nếu chưa tồn tại"""
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        with open(PERFORMANCE_LOG_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'request_id', 'function_name', 'arg_type', 
                'arg_index', 'response_time', 'source', 'prompt_length',
                'response_length', 'status'
            ])
        logger.info(f"Khởi tạo file log hiệu suất: {PERFORMANCE_LOG_FILE}")

# Log thông tin hiệu suất
def log_performance(request_data, response_data, response_time, source, status="success"):
    """Ghi log thông tin hiệu suất vào file CSV"""
    try:
        # Trích xuất thông tin từ prompt
        prompt = request_data.get('prompt', '')
        function_match = re.search(r"Function: ([^\n]+)", prompt)
        type_match = re.search(r"Parameter type: ([^\n]+)", prompt)
        index_match = re.search(r"Parameter index: ([^\n]+)", prompt)
        
        function_name = function_match.group(1) if function_match else "unknown"
        arg_type = type_match.group(1) if type_match else "unknown"
        arg_index = index_match.group(1) if index_match else "-1"
        
        # Tạo request_id từ hash của prompt và timestamp
        request_id = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:8]
        
        # Ghi vào CSV
        with open(PERFORMANCE_LOG_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                datetime.now().isoformat(),
                request_id,
                function_name,
                arg_type,
                arg_index,
                f"{response_time:.3f}",
                source,
                len(prompt),
                len(response_data.get('response', '')) if isinstance(response_data, dict) else 0,
                status
            ])
    except Exception as e:
        logger.error(f"Error logging performance: {e}")

def cache_key(prompt):
    """Tạo key cho cache từ prompt"""
    return hashlib.md5(prompt.encode()).hexdigest()

def add_to_cache(key, value):
    """Thêm kết quả vào cache với LRU logic"""
    with cache_lock:
        # Nếu cache đầy, xóa item cũ nhất
        if len(cache) >= CACHE_SIZE:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        cache[key] = {"value": value, "time": time.time()}
        logger.debug(f"Added to cache: {key[:10]}...")

def get_from_cache(key):
    """Lấy kết quả từ cache nếu có"""
    with cache_lock:
        if key in cache:
            cache[key]["time"] = time.time()  # Cập nhật thời gian truy cập
            logger.debug(f"Cache hit: {key[:10]}...")
            return cache[key]["value"]
    return None

@app.route('/request', methods=['POST'])
def handle_request():
    """
    Xử lý yêu cầu RAG từ fuzzer
    """
    global rag_cache
    
    start_time = time.time()
    request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    
    # Lấy dữ liệu từ request
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({
            "error": "Missing prompt in request", 
            "response": "Không có thông tin phù hợp"
        }), 400
    
    prompt = data['prompt']
    
    # Kiểm tra API key
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set. Please set the environment variable.")
        return jsonify({
            "error": "API key not configured on server", 
            "response": "Server error: API key not configured"
        }), 500
    
    # Kiểm tra cache
    cache_hit = False
    cache_key_value = cache_key(prompt)
    cached_result = get_from_cache(cache_key_value)
    
    if cached_result:
        cache_hit = True
        response_data = {"response": cached_result}
        response_time = time.time() - start_time
        
        # Log performance cho cache hit
        log_performance(data, response_data, response_time, "cache")
        
        return jsonify(response_data)
    
    # Xử lý yêu cầu mới khi không có trong cache
    try:
        # Tạo command để gọi RAG
        command = ["python", RAG_SCRIPT_PATH, "ask", 
                   "--api-key", GOOGLE_API_KEY,
                   "--question", prompt]
        
        # Chạy rag_googleapi.py với timeout
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
            process.wait(timeout=REQUEST_TIMEOUT)
            
            if process.returncode != 0:
                logger.error(f"RAG process error: {stderr.decode('utf-8', errors='replace')}")
                response_data = {"response": "Không có thông tin phù hợp", "error": stderr.decode('utf-8', errors='replace')}
            else:
                output = stdout.decode('utf-8', errors='replace').strip()
                
                # Xử lý output từ RAG
                if "Response from AI:" in output:
                    ai_response = output.split("Response from AI:", 1)[1].strip()
                else:
                    ai_response = output
                
                # Nếu không có kết quả, trả về thông báo mặc định
                if not ai_response or ai_response.isspace():
                    ai_response = "Không có thông tin phù hợp"
                    
                response_data = {"response": ai_response}
                
                # Thêm vào cache nếu thành công
                add_to_cache(cache_key_value, ai_response)
                
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"RAG process timeout after {REQUEST_TIMEOUT} seconds")
            response_data = {"response": "Timeout: Không có phản hồi trong thời gian cho phép", "error": "timeout"}
            
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        response_data = {"response": "Không có thông tin phù hợp do lỗi server", "error": str(e)}
        
    # Tính thời gian phản hồi và log
    response_time = time.time() - start_time
    log_performance(data, response_data, response_time, "rag")
    
    return jsonify(response_data)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint để kiểm tra server còn hoạt động không"""
    return jsonify({"status": "ok", "cache_size": len(cache)})

@app.route('/stats', methods=['GET'])
def get_stats():
    """Endpoint để lấy thông tin thống kê về server"""
    with stats_lock:
        current_stats = stats.copy()
        uptime = time.time() - current_stats["start_time"]
        
        # Tính tỉ lệ
        total = current_stats["total_requests"]
        cache_hit_rate = current_stats["cache_hits"] / total if total > 0 else 0
        success_rate = current_stats["success"] / (total - current_stats["cache_hits"]) if (total - current_stats["cache_hits"]) > 0 else 0
        error_rate = current_stats["errors"] / (total - current_stats["cache_hits"]) if (total - current_stats["cache_hits"]) > 0 else 0
        
        return jsonify({
            "uptime_seconds": uptime,
            "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            "total_requests": total,
            "cache_hits": current_stats["cache_hits"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "success": current_stats["success"],
            "success_rate": f"{success_rate:.2%}",
            "errors": current_stats["errors"],
            "error_rate": f"{error_rate:.2%}",
            "timeouts": current_stats["timeouts"],
            "cache_size": len(cache),
            "cache_limit": CACHE_SIZE
        })

@app.route('/report_vulnerability', methods=['POST'])
def report_vulnerability():
    """
    Nhận báo cáo từ fuzzer khi tìm thấy lỗi để theo dõi hiệu quả của RAG
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Đảm bảo có đủ thông tin cần thiết
    required_fields = ['transaction_id', 'function_name', 'vulnerability_type', 'args']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Chuẩn bị dữ liệu để lưu
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'transaction_id': data['transaction_id'],
        'function_name': data['function_name'],
        'vulnerability_type': data['vulnerability_type'],
        'arguments': json.dumps(data['args']),
        'source': data.get('source', 'unknown'),  # Nguồn tạo ra giá trị (RAG hoặc random)
        'description': data.get('description', '')
    }
    
    # Lưu thông tin lỗi vào file CSV
    file_exists = os.path.exists(VULNERABILITY_LOG_FILE)
    with open(VULNERABILITY_LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    
    logger.info(f"Recorded vulnerability: {data['vulnerability_type']} in {data['function_name']}")
    
    return jsonify({"status": "success", "message": "Vulnerability recorded"})

if __name__ == "__main__":
    # Khởi tạo file log hiệu suất
    init_performance_log()
    
    logger.info(f"Starting RAG Server on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

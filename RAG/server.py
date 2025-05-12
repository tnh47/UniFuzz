from flask import Flask, request, jsonify
import os
import subprocess
import logging
import json
import hashlib
from functools import lru_cache
import threading
import time

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
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyB3P2COlotMu-3RR-ehwZXZk60wOWJvfEA")
REQUEST_TIMEOUT = 120  # seconds
CACHE_SIZE = 100  # Số lượng kết quả lớn nhất lưu trong cache

# Cache kết quả để tránh gọi lại RAG nhiều lần cho cùng prompt
cache = {}
cache_lock = threading.Lock()

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
    """Xử lý request từ llm_agent"""
    try:
        # Lấy prompt từ JSON body
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing prompt in request body"}), 400

        prompt = data['prompt']
        prompt_hash = cache_key(prompt)
        logger.info(f"Received prompt ({len(prompt)} chars), hash: {prompt_hash[:10]}...")
        
        # Kiểm tra cache trước
        cached_response = get_from_cache(prompt_hash)
        if cached_response:
            logger.info(f"Returning cached response for {prompt_hash[:10]}...")
            return jsonify({"response": cached_response, "source": "cache"})

        # Log full prompt for debugging
        logger.debug(f"Full prompt: {prompt}")
        
        # Gọi RAG script với timeout tăng lên
        try:
            result = subprocess.run(
                [
                    "python3.12",  # Đảm bảo sử dụng Python 3.12
                    RAG_SCRIPT_PATH,
                    "ask",
                    "--api-key", GOOGLE_API_KEY,
                    "--question", prompt
                ],
                capture_output=True,
                text=True,
                timeout=REQUEST_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            logger.error(f"RAG execution timed out after {REQUEST_TIMEOUT}s")
            return jsonify({"error": "RAG execution timed out", "fallback": "use_random"}), 504

        # Xử lý output chi tiết hơn
        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode != 0:
            logger.error(f"RAG execution failed (code {result.returncode}): {error}")
            return jsonify({"error": "RAG execution failed", "details": error, "fallback": "use_random"}), 500

        # Xử lý response
        if "Response from AI:" in output:
            output = output.split("Response from AI:", 1)[1].strip()
        print("\n" + "="*50)
        print(f"PROMPT: {prompt[:100]}...")
        print(f"RAG RESPONSE: {output}")
        print("="*50 + "\n")
        # Loại bỏ các ký tự không mong muốn và format
        output = output.strip()
        logger.info(f"RAG response ({len(output)} chars): {output[:50]}...")
        
        # Lưu vào cache
        add_to_cache(prompt_hash, output)
        
        return jsonify({"response": output, "source": "rag"})

    except Exception as e:
        logger.exception(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e), "fallback": "use_random"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint để kiểm tra server còn hoạt động không"""
    return jsonify({"status": "ok", "cache_size": len(cache)})

@app.route('/stats', methods=['GET'])
def stats():
    """Endpoint để lấy thông tin về server"""
    return jsonify({
        "cache_size": len(cache),
        "cache_limit": CACHE_SIZE
    })

if __name__ == "__main__":
    logger.info(f"Starting RAG Server on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

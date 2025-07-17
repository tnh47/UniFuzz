from flask import Flask, request, jsonify
import os
import subprocess
import logging
import json
import hashlib
import threading
import time
import csv
from datetime import datetime
import re

app = Flask(__name__)

# Logging to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAGServer")

# Configuration
RAG_SCRIPT_PATH = "RAG/rag_googleapi.py"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
REQUEST_TIMEOUT = 120        # seconds
CACHE_SIZE = 100             # max cache entries
PERFORMANCE_LOG_FILE = "rag_performance.csv"
VULNERABILITY_LOG_FILE = "rag_vulnerabilities.csv"

# In-memory LRU cache
cache = {}
cache_lock = threading.Lock()

# Stats for monitoring
stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "timeouts": 0,
    "errors": 0,
    "success": 0,
    "start_time": time.time()
}
stats_lock = threading.Lock()

def init_performance_log():
    """Initialize performance CSV if missing."""
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        with open(PERFORMANCE_LOG_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'request_id', 'function_name', 'arg_type', 
                'arg_index', 'response_time', 'source', 'prompt_length',
                'response_length', 'status'
            ])
        logger.info(f"Created performance log: {PERFORMANCE_LOG_FILE}")

def log_performance(request_data, response_data, response_time, source, status="success"):
    """Append performance metrics to CSV."""
    try:
        prompt = request_data.get('prompt', '')
        # Extract metadata from prompt if available
        function_match = re.search(r"Function: ([^\n]+)", prompt)
        type_match     = re.search(r"Parameter type: ([^\n]+)", prompt)
        index_match    = re.search(r"Parameter index: ([^\n]+)", prompt)
        
        function_name = function_match.group(1) if function_match else "unknown"
        arg_type      = type_match.group(1)     if type_match     else "unknown"
        arg_index     = index_match.group(1)    if index_match    else "-1"
        
        request_id = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:8]
        
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
    return hashlib.md5(prompt.encode()).hexdigest()

def add_to_cache(key, value):
    with cache_lock:
        if len(cache) >= CACHE_SIZE:
            oldest = next(iter(cache))
            del cache[oldest]
        cache[key] = {"value": value, "time": time.time()}

def get_from_cache(key):
    with cache_lock:
        if key in cache:
            cache[key]["time"] = time.time()
            return cache[key]["value"]
    return None

@app.route('/request', methods=['POST'])
def handle_request():
    start_time = time.time()
    stats["total_requests"] += 1

    data = request.json or {}
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Missing prompt", "response": "No relevant information available"}), 400

    # Print prompt to terminal
    print("\n==================== REQUEST RECEIVED ====================")
    print(f"Prompt sent to AI (length {len(prompt)} characters):\n{prompt}")
    print("========================================================\n")

    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not configured")
        return jsonify({"error": "API key missing", "response": "Server error"}), 500
    
    key = cache_key(prompt)
    cached = get_from_cache(key)
    if cached:
        stats["cache_hits"] += 1
        resp = {"response": cached}
        log_performance(data, resp, time.time() - start_time, "cache")
        # Print response to terminal
        print("-------------------- RESPONSE (CACHE) -------------------")
        print(f"Value returned from cache:\n{cached}")
        print("========================================================\n")
        return jsonify(resp)

    try:
        cmd = [
            "python", RAG_SCRIPT_PATH, "ask",
                   "--api-key", GOOGLE_API_KEY,
            "--question", prompt
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = proc.communicate(timeout=REQUEST_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            stats["timeouts"] += 1
            logger.error("RAG process timeout")
            resp = {"response": "Timeout: No response received within the allowed time", "error": "timeout"}
            log_performance(data, resp, time.time() - start_time, "rag", status="timeout")
            print("-------------------- RESPONSE (TIMEOUT) -----------------")
            print("Timeout or no response received within the allowed time.")
            print("========================================================\n")
            return jsonify(resp)

        if proc.returncode != 0:
            stats["errors"] += 1
            error_msg = stderr.decode(errors='ignore')
            logger.error(f"RAG error: {error_msg}")
            resp = {"response": "No relevant information found", "error": error_msg}
            print("-------------------- RESPONSE (ERROR) -------------------")
            print(f"Error when calling AI:\n{error_msg}")
            print("========================================================\n")
        else:
            output = stdout.decode(errors='ignore').strip()
            if "Response from AI:" in output:
                ai_resp = output.split("Response from AI:",1)[1].strip()
            else:
                ai_resp = output or "No relevant information found"
            resp = {"response": ai_resp}
            add_to_cache(key, ai_resp)
            stats["success"] += 1
            print("-------------------- RESPONSE (AI) ----------------------")
            print(f"Value returned from AI:\n{ai_resp}")
            print("========================================================\n")
            
    except Exception as e:
        stats["errors"] += 1
        logger.error(f"Server processing error: {e}")
        resp = {"response": "No relevant information due to server error", "error": str(e)}
        print("-------------------- RESPONSE (EXCEPTION) ---------------")
        print(f"Server error: {e}")
        print("========================================================\n")

    log_performance(data, resp, time.time() - start_time, "rag")
    return jsonify(resp)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "cache_size": len(cache)})

@app.route('/stats', methods=['GET'])
def get_stats():
    with stats_lock:
        uptime = time.time() - stats["start_time"]
        total = stats["total_requests"]
        cache_hit_rate = stats["cache_hits"] / total if total else 0
        success_rate   = stats["success"] / (total - stats["cache_hits"]) if (total - stats["cache_hits"]) else 0
        error_rate     = stats["errors"] / (total - stats["cache_hits"]) if (total - stats["cache_hits"]) else 0
        
        return jsonify({
            "uptime_seconds": uptime,
            "total_requests": total,
            "cache_hits": stats["cache_hits"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "success": stats["success"],
            "success_rate": f"{success_rate:.2%}",
            "errors": stats["errors"],
            "error_rate": f"{error_rate:.2%}",
            "timeouts": stats["timeouts"],
            "cache_size": len(cache),
            "cache_limit": CACHE_SIZE
        })

@app.route('/report_vulnerability', methods=['POST'])
def report_vulnerability():
    data = request.json or {}
    required = ['transaction_id', 'function_name', 'vulnerability_type', 'args']
    for f in required:
        if f not in data:
            return jsonify({"error": f"Missing {f}"}), 400
    
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'transaction_id': data['transaction_id'],
        'function_name': data['function_name'],
        'vulnerability_type': data['vulnerability_type'],
        'arguments': json.dumps(data['args']),
        'source': data.get('source','unknown'),
        'description': data.get('description','')
    }
    
    exists = os.path.exists(VULNERABILITY_LOG_FILE)
    with open(VULNERABILITY_LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(log_data)
    
    logger.info(f"Recorded vulnerability: {data['vulnerability_type']} in {data['function_name']}")
    return jsonify({"status":"success"})

if __name__ == "__main__":
    init_performance_log()
    logger.info("Starting RAG Server on port 5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)

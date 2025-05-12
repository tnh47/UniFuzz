# UniFuzz
source venv3.8/bin/activate  # Trên Linux/MacOS
pip install -r requirements.txt
## De su dung python 3.12
source .venv/bin/activate  # Python 3.12
## Fuzzing binh thuong
python fuzzer/main.py --source ./SmartContract/ABC.sol --solc-path-cross /home/ngonhat/Desktop/UniFuzz/venv3.8/bin/solc --contract ABC --solc v0.8.26 --api-key "AIzaSyB3P2COlotMu-3RR-ehwZXZk60wOWJvfEA"
## Su dung llm
python fuzzer/main.py --source ./SmartContract/ABC.sol --solc-path-cross /home/ngonhat/Desktop/UniFuzz/venv3.8/bin/solc --contract ABC --solc v0.8.26 --api-key "AIzaSyB3P2COlotMu-3RR-ehwZXZk60wOWJvfEA" --use-llm
### file generator.py la cua CrossFuzz
### 2 file llm_agent va llm_enhanced_generator la 2 file them vao 
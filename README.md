# UniFuzz
![UniFuzz drawio-1](https://github.com/user-attachments/assets/25a69a22-1ffb-4b45-8df9-7eadfec9151b)
python -m venv venv
source venv/bin/activate  # Trên Linux/MacOS
venv\Scripts\activate  # Trên Windows
pip install -r requirements.txt
python main.py learn data/raw_pdfs/your_contract.pdf
python main.py query "Làm thế nào phát hiện lỗi reentrancy?"
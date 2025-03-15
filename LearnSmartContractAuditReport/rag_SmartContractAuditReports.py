# -*- coding: utf-8 -*-
import torch
import ollama
import os
import argparse
import json
import re
from pypdf import PdfReader

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Extract text from PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Extract common vulnerabilities from the report content
def extract_vulnerabilities(text):
    vulnerabilities = []
    patterns = [
        r"Reentrancy",
        r"Integer Overflow",
        r"Phishing Attack",
        r"Unauthorized Access",
        r"Timestamp Manipulation",
        r"Unchecked External Calls",
        r"Missing Events for Significant Transactions"
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            vulnerabilities.append(pattern)
    return vulnerabilities

# Generate testing guidance using LLM
def generate_testing_guidance(text, ollama_model):
    prompt = f"""
    Based on the following smart contract audit report, extract:
    1. Common security vulnerabilities detected.
    2. Special testing conditions required.
    3. Critical logic or invariants that need verification.
    
    Audit report content:
    {text}
    """
    response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# CLI processing
def main():
    parser = argparse.ArgumentParser(description="Analyze smart contract audit reports using Ollama LLM")
    parser.add_argument("--pdf", dest="pdf_path", required=True, help="Path to the PDF audit report")
    parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
    args = parser.parse_args()

    # Check if PDF file exists
    if not os.path.exists(args.pdf_path):
        print("PDF file not found.")
        return

    print("\nAnalyzing audit report...")
    text = extract_text_from_pdf(args.pdf_path)

    # Detect common vulnerabilities
    vulnerabilities = extract_vulnerabilities(text)
    print(f"\nDetected vulnerabilities: {', '.join(vulnerabilities) if vulnerabilities else 'None found'}")

    # Generate testing guidance
    print("\nGenerating testing guidance...")
    guidance = generate_testing_guidance(text, args.model)
    print(f"\n=> Testing Guidance:\n{guidance}\n")

if __name__ == "__main__":
    main()

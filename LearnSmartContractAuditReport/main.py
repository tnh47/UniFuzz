# main.py
import streamlit as st
from fuzzing import fuzz_smart_contract

def main():
    st.title("Smart Contract Analyzer and Fuzzer")
    st.write("Upload a smart contract audit report (PDF) to analyze and fuzz.")

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("Analyzing audit report...")

        # Phân tích và fuzz
        guidance = fuzz_smart_contract(uploaded_file, "llama3")
        st.write("### Analysis and Fuzzing Guidance")
        st.write(guidance)

if __name__ == "__main__":
    main()
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from app.config import Config
from app.pdf_processor import PDFProcessor
from app.vector_store import VectorStore
from app.llm_handler import GeminiHandler # Changed import
from app.fuzzer_integration import Fuzzer
from app.audit_agent import AuditAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def learn_command(pdf_path: Path):
    """Optimized learning workflow for PDF documents - Processes PDF, creates embeddings, and saves to vector store."""
    try:
        processor = PDFProcessor()
        vs = VectorStore()

        with tqdm(total=3, desc="Processing Pipeline") as pbar:
            pbar.set_description("Processing PDF content")
            chunks = processor.process_pdf(pdf_path)
            pbar.update(1)

            if not chunks:
                logging.warning(f"No content chunks extracted from {pdf_path.name}. Learning process skipped.")
                return

            pbar.set_description("Creating embeddings and adding to vector store")
            vs.add_documents(chunks)
            pbar.update(1)

            pbar.set_description("Saving vector index and metadata")
            vs.save_index()
            pbar.update(1)

        logging.info(f"Successfully learned from audit report: {pdf_path.name}")

    except Exception as e:
        logging.error(f"Learning process failed for {pdf_path.name}: {str(e)}", exc_info=True) # Log full exception info
        exit(1)


def query_command(question: str):
    """Query the vector store and get answer using LLM -  Searches vector store, gets context, and generates answer."""
    try:
        audit_agent = AuditAgent() # Use AuditAgent for query and potential fuzzing
        response = audit_agent.analyze_audit_report(question) # Analyze report using AuditAgent

        print("\n🔍 **Question:**", question) # Markdown for bold question
        print("=" * 60)
        print(f"📝 **Answer:**\n{response}\n") # Markdown for bold answer
        print("=" * 60)
        # Source context is already included in the formatted response from VectorStore in audit_agent now.
        # No need to print context separately here unless you want to.
        # print("📚 **Reference Sources:**")
        # print(context) # Context is now part of the response from audit_agent

    except Exception as e:
        logging.error(f"Query failed: {str(e)}", exc_info=True) # Log full exception info
        exit(1)

def fuzz_command(target: str):
    """Run fuzzing tests on a target contract - Standalone fuzzing command."""
    try:
        fuzzer = Fuzzer()
        results = fuzzer.run_fuzzing(Config.DEFAULT_FUZZING_TOOL, Path(target))

        print("\n🔧 **Fuzzing Results:**") # Markdown for bold section title
        print("=" * 60)
        print(f"**Tool:** {results['tool']}") # Bold tool name
        print(f"**Success:** {results['success']}") # Bold success status
        print("\n**Output:**") # Bold output label
        print(results['output'])
        print("\n**Errors:**") # Bold errors label
        print(results['errors'])

        if not results['success']:
            logging.warning("Fuzzing was not completely successful. Review output and errors.")
        else:
            logging.info("Fuzzing completed successfully (according to tool's return code). Review output for findings.")


    except Exception as e:
        logging.error(f"Fuzzing failed: {str(e)}", exc_info=True) # Log full exception info
        exit(1)


def main():
    """Main CLI entry point - Parses arguments and executes commands."""
    Config.setup_directories() # Ensure directories are set up

    parser = argparse.ArgumentParser(
        description="Smart Contract Audit Analysis System - Learn from reports, query for insights, and run fuzzing.",
        formatter_class=argparse.RawTextHelpFormatter # Keep raw text formatter for help messages
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # 'learn' command parser
    learn_parser = subparsers.add_parser('learn', help='Process and learn from a PDF audit report.')
    learn_parser.add_argument('pdf_file', type=str, help='Path to the PDF audit report file.')

    # 'query' command parser
    query_parser = subparsers.add_parser('query', help='Query the audit reports with a question.')
    query_parser.add_argument('question', type=str, help='The question to ask about the audit reports.')

    # 'fuzz' command parser - Clarified help message
    fuzz_parser = subparsers.add_parser('fuzz', help='Run standalone fuzzing tests on a Solidity contract. \nNote: This is a direct fuzzing command, not related to audit report analysis (yet).\nFor audit-report-driven fuzzing, use the \'query\' command which may trigger automated fuzzing based on analysis.')
    fuzz_parser.add_argument('target', type=str, help='Path to the Solidity contract file to fuzz.')


    args = parser.parse_args()

    try:
        if args.command == 'learn':
            learn_command(Path(args.pdf_file))
        elif args.command == 'query':
            query_command(args.question)
        elif args.command == 'fuzz':
            fuzz_command(args.target)
        elif args.command is None: # No command given, print help
            parser.print_help()
        else:
            logging.error(f"Unknown command: {args.command}")
            parser.print_help() # Show help for unknown commands
            exit(1)


    except SystemExit as e: # Catch SystemExit from commands (e.g., from learn_command on failure)
        exit(e.code)
    except Exception as e: # Catch any unexpected exceptions at the top level
        logging.critical(f"Unhandled system error: {str(e)}", exc_info=True) # Log critical error with traceback
        print(f"\n🚨 **Critical Error:** An unexpected error occurred. Check the logs for details.") # User-friendly error message
        exit(1)

if __name__ == "__main__":
    main()


"""
**To run this application:**

1.  **Set Environment Variables:**
    -   `GEMINI_API_KEY`: Your Google Gemini API key.
    -   Optionally, you can set other configuration variables like `CHUNK_SIZE`, `VECTOR_DB_DIR`, etc., as environment variables to override defaults in `config.py`.

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt  # Create a requirements.txt with: fitz openai faiss-cpu requests tqdm tenacity google-generativeai
    # Also ensure you have fuzzing tools like Echidna or Foundry installed separately if you plan to use the 'fuzz' command.
    ```

3.  **Run Commands:**

    **Learn from a PDF audit report:**
    ```bash
    python main.py learn path/to/your/audit_report.pdf
    ```

    **Query the audit reports:**
    ```bash
    python main.py query "What are the critical vulnerabilities reported in the audit?"
    ```

    **Run standalone fuzzing (on a Solidity file):**
    ```bash
    python main.py fuzz path/to/your/ContractTest.sol
    ```

**To improve this application further (as discussed):**

*   **Implement more sophisticated fuzz test generation** in `fuzzer_integration.py` based on vulnerability types from the fuzzing plan.
*   **Enhance keyword search and RRF reranking** in `vector_store.py` for better hybrid search results.
*   **Expand PDF section detection** and metadata extraction in `pdf_processor.py` to handle more audit report formats.
    *   **Add more fuzzing tools** to `fuzzer_integration.py` and make tool selection more flexible.
    *   **Create a comprehensive README.md** explaining setup, usage, configuration, architecture, and further development.
    *   **Add unit tests and integration tests** for core components.
    *   **Consider a web interface or API** for easier user interaction.
"""
from app.vector_store import VectorStore
from app.llm_handler import GeminiHandler  # Changed import
from app.fuzzer_integration import Fuzzer
from app.config import Config
from pathlib import Path

class AuditAgent:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = GeminiHandler() # Changed to GeminiHandler
        self.fuzzer = Fuzzer()

    def analyze_audit_report(self, question: str):
        # Retrieve relevant context
        context = self.vector_store.hybrid_search(question)

        # Generate initial response
        response = self.llm.generate_response(context, question)

        # Automatic fuzzing trigger - now uses LLM and config triggers
        if self._should_trigger_fuzzing(response, context):
            fuzzing_plan = self.llm.generate_fuzzing_plan(context)
            if fuzzing_plan: # Proceed only if a plan is generated
                test_file = self._generate_tests(fuzzing_plan)
                fuzz_results = self.fuzzer.run_fuzzing(
                    Config.DEFAULT_FUZZING_TOOL,
                    test_file
                )
                response += "\n\n**Fuzzing Results:**\n" + self._format_fuzz_results(fuzz_results)
            else:
                response += "\n\n**Fuzzing Plan Generation Failed.** Could not create a fuzzing plan."

        return response

    def _should_trigger_fuzzing(self, response: str, context: str) -> bool:
        """Determine if fuzzing should be triggered - now with LLM check and config triggers"""
        # Check for keywords in the response (from config)
        keyword_trigger = any(keyword in response.lower() for keyword in Config.FUZZING_TRIGGER_KEYWORDS)
        if keyword_trigger:
            return True

        # Optional: LLM-based trigger (more intelligent but slower and API cost)
        if Config.LLM_FUZZING_TRIGGER_ENABLED:
            llm_trigger_prompt = f"""
            Audit Report Analysis Response: {response}

            Based on the response above and the context of a smart contract audit report,
            is fuzzing recommended to further investigate potential vulnerabilities? Answer 'yes' or 'no'."""
            llm_trigger_response = self.llm._call_llm(llm_trigger_prompt, temperature=0.1) # Lower temp for deterministic yes/no
            return "yes" in llm_trigger_response.lower()
        return False # Default to no fuzzing if no keywords and LLM trigger disabled

    def _generate_tests(self, plan: dict) -> Path:
        """Generate test files from fuzzing plan"""
        test_content = self.fuzzer.generate_fuzz_tests(plan.get("vulnerabilities", []))
        test_file = Config.FUZZING_WORKSPACE / "fuzz_test.sol"
        test_file.write_text(test_content)
        return test_file

    def _format_fuzz_results(self, fuzz_results: dict) -> str:
        """Format fuzzing results for user display"""
        output = f"Tool: {fuzz_results['tool']}\n"
        output += f"Success: {fuzz_results['success']}\n"

        if fuzz_results['success']:
            output += "**Fuzzing Output:**\n"
            # Summarize key findings if possible - simple keyword search for now
            findings_keywords = ["vulnerability", "crash", "assertion", "fail"]
            relevant_output_lines = [
                line for line in fuzz_results['output'].splitlines()
                if any(keyword in line.lower() for keyword in findings_keywords)
            ]
            if relevant_output_lines:
                output += "\n".join(relevant_output_lines) + "\n"
            else:
                output += "No immediate vulnerabilities detected in summarized output.\n"

        else:
            output += "**Fuzzing Errors:**\n"
            output += fuzz_results['errors']

        return output
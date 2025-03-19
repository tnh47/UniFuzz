from typing import List, Dict
import requests
import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import Config
import google.generativeai as genai # Import google-generativeai

logger = logging.getLogger(__name__)

class GeminiHandler: # Renamed from DeepseekHandler to GeminiHandler
    def __init__(self):
        genai.configure(api_key=Config.get_gemini_api_key()) # Initialize Gemini API client


    def _call_llm(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str: # Internal method to call LLM with retry
        """Internal method to call Gemini API with retry logic using google-generativeai library"""
        try:
            model = genai.GenerativeModel(Config.GEMINI_TEXT_MODEL) # Load text generation model

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig( # GenerationConfig for temperature and max_tokens
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            if not response.candidates: # Handle no candidates in response
                raise ValueError("No response candidates returned from Gemini API")

            return response.candidates[0].content.parts[0].text.strip() # Extract text from response


        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            return "Error communicating with LLM. Please try again later." # More user-friendly error message


    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=20),
           before_sleep=lambda _: logger.warning("Retrying LLM request..."))
    def generate_response(self, context: str, question: str) -> str:
        """Generate context-aware response for user queries using Gemini API"""
        try:
            prompt = self._build_messages(context, question) # Build prompt using context and question
            response_text = self._call_llm(prompt) # Call Gemini API using the built prompt
            return response_text

        except Exception as e:
            logger.error(f"Gemini API failed: {str(e)}")
            return "Currently unable to process request. Please try again later." # More user-friendly error

    def _build_messages(self, context: str, question: str) -> str: # Modified to return a single prompt string for Gemini
        """Construct prompt for Gemini API - now returns a single string"""
        return f"""Bạn là chuyên gia phân tích hợp đồng thông minh.
Chỉ sử dụng thông tin từ context được cung cấp.

Context:\n{context}\n\nCâu hỏi: {question}\n\n
Yêu cầu:\n- Trả lời chính xác, rõ ràng\n
- Nếu không đủ thông tin, hãy nói 'Không tìm thấy thông tin'\n
- Đánh dấu các điều khoản quan trọng bằng **bold**\n
- Trả lời ngắn gọn và đi thẳng vào vấn đề.""" # Combined prompt as a single string for Gemini


    def generate_fuzzing_plan(self, context: str) -> dict or None: # Return None on failure
        """Generate fuzzing test plan from audit report using Gemini API - Improved prompt and error handling"""
        prompt = f"""Phân tích báo cáo audit sau và tạo kế hoạch fuzzing:
{context}

Yêu cầu:
- Liệt kê các lỗ hổng cần kiểm tra
- Đề xuất loại fuzzing phù hợp cho từng lỗ hổng
- Tạo các test case mẫu
- Định dạng đầu ra JSON"""

        response = self._call_llm(prompt, temperature=0.2, max_tokens=1500) # Slightly lower temp for plan, increased max tokens
        return self._parse_json_response(response) # Assuming JSON parsing still needed

    def _parse_json_response(self, response: str) -> dict or None:
        """Safe JSON parsing - Now returns None on error instead of empty dict"""
        try:
            parsed_json = json.loads(response)

            # Basic validation - check if it's a dict and has 'vulnerabilities' key (or is empty dict)
            if not isinstance(parsed_json, dict):
                raise ValueError("Response is not a JSON object") # More specific error
            if parsed_json and 'vulnerabilities' in parsed_json and not isinstance(parsed_json['vulnerabilities'], list): # Check vulnerabilities list if dict is not empty
                raise ValueError("'vulnerabilities' field is not a list")

            logger.info("Successfully parsed JSON response from LLM.")
            return parsed_json

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM: JSONDecodeError: {e}")
            logger.error(f"Response text was: {response}") # Log the problematic response
            return None # Return None on parsing failure

        except ValueError as ve: # Catch validation errors
            logger.error(f"Invalid JSON format from LLM: {ve}")
            logger.error(f"Response text was: {response}") # Log the problematic response
            return None # Return None on validation failure

        except Exception as e: # Catch any other unexpected errors during parsing
            logger.error(f"Unexpected error parsing JSON response: {e}")
            logger.error(f"Response text was: {response}") # Log the problematic response
            return None # Return None on unexpected error
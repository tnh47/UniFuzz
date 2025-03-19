import google.generativeai as genai
import numpy as np
import logging
import json # Import json for response handling
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from app.config import Config

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self):
        self.successful_calls = 0
        self.failed_calls = 0
        genai.configure(api_key=Config.get_gemini_api_key()) # Initialize Gemini API client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((
            Exception, # Catch broad exceptions for retry with google-generativeai
            ValueError # Include ValueError in retry for potential API errors in response parsing
        )),
        before_sleep=lambda _: logger.warning("Retrying API connection...") # More descriptive retry message
    )
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Handle API calls to Google Gemini Embedding API using google-generativeai library"""
        try:
            if not texts or len(texts) > 100: # Basic batch size validation
                raise ValueError("Batch size must be between 1 and 100")

            model = genai.GenerativeModel(Config.GEMINI_EMBEDDING_MODEL) # Load embedding model

            response = model.embed(content=texts, task_type="retrieval_query") # Embed content

            # Extract embeddings from Gemini API response format
            embeddings = []
            for embedding_data in response.result.embeddings: # Access embeddings via response.result.embeddings
                embeddings.append(embedding_data.values) # Access embedding values via embedding_data.values


            self.successful_calls += 1
            return embeddings

        except Exception as e: # Catch broad exceptions with google-generativeai
            self.failed_calls += 1
            logger.error(f"Gemini API request failed: {str(e)}")
            raise # Re-raise to trigger retry

    def batch_process(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Dynamic batch processing with size adjustment - more robust batching"""
        embeddings = []
        with tqdm(total=len(texts), desc="Creating embeddings") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    embeds = self.get_embeddings(batch)
                    embeddings.extend(embeds)
                    # Dynamic batch size adjustment - more conservative adjustment
                    if self.failed_calls > 0: # Reduce batch size if failures occurred
                        batch_size = max(16, batch_size // 2) # Reduce more aggressively, but min 16
                        self.failed_calls = 0 # Reset failed count after successful batch
                    else: # Increase batch size if successful
                        batch_size = min(100, batch_size * 2) # Increase more aggressively, max 100 (API limit - check Gemini API docs for limits)

                except Exception as e:
                    logger.warning(f"Batch processing encountered error: {e}. Reducing batch size to {batch_size} and retrying...")
                    batch_size = max(16, batch_size // 2) # Reduce batch size on error, min 16 - try smaller batches
                    self.failed_calls += 1 # Increment failed calls to trigger more aggressive reduction next time (if consecutive failures)
                    # No need to re-raise, batch_process continues with reduced batch size

                finally:
                    pbar.update(len(batch))
        return embeddings
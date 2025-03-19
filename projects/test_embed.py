import google.generativeai as genai
from app.config import Config  # Assuming config.py is in the 'app' directory

genai.configure(api_key=Config.get_gemini_api_key())

model = genai.GenerativeModel(Config.GEMINI_EMBEDDING_MODEL) # Use model name from your config

texts = ["This is a test sentence.", "Another test."]

try:
    response = model.embed(content=texts, task_type="retrieval_query")
    print("Embeddings generated successfully:")
    for embedding in response.result.embeddings:
        print(embedding.values[:5]) # Print first 5 dimensions of each embedding
except Exception as e:
    print(f"Error: {e}")
import faiss
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from app.config import Config
from app.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

# Custom exceptions
class DimensionMismatchError(Exception): pass
class InvalidMetadataError(Exception): pass

class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []
        self.embedding_size = Config.GEMINI_EMBEDDING_DIMENSION  # Dimension for Gemini Embedding model
        self._init_storage()
        self._validate_embedding_dimension()

    def _validate_embedding_dimension(self):
        if Config.GEMINI_EMBEDDING_DIMENSION != 768: # Validate against expected Gemini dimension (768)
            raise ValueError(f"Embedding dimension mismatch in config: Expected 768, but got {Config.GEMINI_EMBEDDING_DIMENSION}")

    def hybrid_search(self, query: str, top_k: int = 5, keyword_weight: float = 0.3) -> List[Dict]: # Added keyword_weight
        """Combine semantic and keyword search - Rerank with Reciprocal Rank Fusion"""
        semantic_results = self._semantic_search(query, top_k)
        keyword_results = self._keyword_search(query, top_k)
        return self._rerank_results(semantic_results, keyword_results, keyword_weight=keyword_weight, top_k=top_k) # Rerank with RRF

    def _init_storage(self):
        """Initialize FAISS index and load metadata from disk - Handles corrupted storage"""
        index_file = Config.VECTOR_DB_DIR / "faiss_index.index"
        metadata_file = Config.VECTOR_DB_DIR / "metadata.json"

        try:
            if index_file.exists() and metadata_file.exists(): # Check both files exist
                self.index = faiss.read_index(str(index_file))
                if self.index.d != self.embedding_size:
                    raise DimensionMismatchError(
                        f"Embedding dimension mismatch: Index {self.index.d} vs Config {self.embedding_size}"
                    )
                with open(metadata_file, 'r') as f:
                    raw_metadata = json.load(f)
                    if not isinstance(raw_metadata, list) or any(not isinstance(m, dict) for m in raw_metadata):
                        raise InvalidMetadataError("Invalid metadata structure in JSON file")
                    self.metadata = raw_metadata
                logger.info(f"Loaded vector store from disk: {self.index.ntotal} entries")

            else: # Create new index if files don't exist
                self.index = faiss.IndexFlatL2(self.embedding_size)
                self.metadata = [] # Initialize empty metadata list
                logger.info("Created new vector store in memory.")

        except (DimensionMismatchError, InvalidMetadataError) as e:
            logger.error(f"Vector storage corrupted: {str(e)} - Resetting storage...")
            self._reset_storage()

        except Exception as e: # Catch any other potential loading errors
            logger.error(f"Error initializing vector store: {str(e)} - Resetting storage...", exc_info=True)
            self._reset_storage()


    def _reset_storage(self):
        """Reset storage to a clean state - Deletes index and metadata files and re-inits"""
        index_file = Config.VECTOR_DB_DIR / "faiss_index.index"
        metadata_file = Config.VECTOR_DB_DIR / "metadata.json"

        try: # Use try-except to handle cases where files might not exist
            index_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            logger.warning("Deleted corrupted vector store files.")
        except Exception as e:
            logger.warning(f"Error deleting vector store files during reset: {e}") # Non-critical error during reset

        self.index = faiss.IndexFlatL2(self.embedding_size) # Re-initialize index
        self.metadata = [] # Clear metadata
        logger.info("Vector storage reset to a clean state.")


    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to vector store - Validates chunks and embeddings"""
        if not chunks:
            logger.warning("No chunks provided to add_documents. Skipping.")
            return

        try:
            # Validate input chunks structure
            if any('text' not in c or 'metadata' not in c for c in chunks):
                raise ValueError("Invalid chunk structure: Each chunk must have 'text' and 'metadata' keys.")

            texts = [c['text'] for c in chunks]
            metadatas = [c['metadata'] for c in chunks] # Keep all metadata for saving

            # Generate embeddings in batch
            embedder = EmbeddingManager()
            embeddings = embedder.batch_process(texts)

            # Dimension and count validation after embedding generation
            if not embeddings:
                raise ValueError("No embeddings generated.")
            if len(embeddings) != len(texts):
                raise ValueError(f"Number of embeddings does not match number of texts: {len(embeddings)} vs {len(texts)}")
            if len(embeddings[0]) != self.embedding_size:
                raise DimensionMismatchError(f"Embedding dimension mismatch: Expected {self.embedding_size}, got {len(embeddings[0])}")

            vectors = np.array(embeddings).astype('float32') # Convert embeddings to numpy array

            self.index.add(vectors) # Add vectors to FAISS index
            self.metadata.extend(metadatas) # Add metadata to list

            logger.info(f"Added {len(chunks)} documents to vector store. Current index size: {self.index.ntotal}")

        except DimensionMismatchError as e:
            logger.error(f"Dimension mismatch during add_documents: {e}")
            self._recover_from_failure() # Attempt recovery
            raise # Re-raise exception after recovery attempt

        except Exception as e: # Catch any other errors during document adding
            logger.error(f"Error adding documents to vector store: {str(e)}", exc_info=True) # Log full exception info
            self._recover_from_failure() # Attempt recovery
            raise # Re-raise exception after recovery attempt


    def save_index(self):
        """Save FAISS index and metadata to disk - Performs validation before saving"""
        index_file = Config.VECTOR_DB_DIR / "faiss_index.index"
        metadata_file = Config.VECTOR_DB_DIR / "metadata.json"

        try:
            if self.index.ntotal != len(self.metadata): # Critical validation: index and metadata count must match
                raise ValueError(f"Index size and metadata count mismatch: Index size = {self.index.ntotal}, Metadata count = {len(self.metadata)}. Data integrity compromised.")

            index_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists before saving

            faiss.write_index(self.index, str(index_file)) # Save FAISS index
            with open(metadata_file, 'w', encoding='utf-8') as f: # Save metadata to JSON file with UTF-8 encoding
                json.dump(self.metadata, f, ensure_ascii=False, indent=2) # Ensure_ascii=False for Unicode chars, indent for readability

            logger.info(f"Saved vector store index ({self.index.ntotal} entries) and metadata to disk.")

        except ValueError as ve:
            logger.error(f"Data integrity error during save_index: {ve} - Data not saved.")

        except Exception as e: # Catch any other potential save errors
            logger.error(f"Error saving vector store index and metadata to disk: {str(e)}", exc_info=True)


    def search_with_context(self, query: str, top_k: int = 5) -> str:
        """Search and format results with context - User-friendly output"""
        results = self.hybrid_search(query, top_k=top_k) # Use hybrid search
        return self._format_results(results) if results else "No relevant information found in the audit reports." # User-friendly no results message

    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform semantic search using FAISS index - Returns list of result dictionaries"""
        try:
            embedder = EmbeddingManager()
            query_embedding = embedder.get_embeddings([query])[0] # Get embedding for query

            distances, indices = self.index.search(
                np.array([query_embedding]).astype('float32'),
                top_k
            )

            results = []
            for distance, index in zip(distances[0], indices[0]):
                if index != -1: # Check for valid index (-1 means no result)
                    metadata_with_score = self.metadata[index].copy() # Get metadata from index
                    metadata_with_score['semantic_distance'] = distance # Add semantic distance to metadata
                    metadata_with_score['semantic_score'] = 1.0 / (1.0 + distance) # Convert distance to score (higher is better)
                    results.append(metadata_with_score)
            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}", exc_info=True)
            return [] # Return empty list on error


    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Basic keyword search in metadata text - Returns list of metadata dictionaries sorted by keyword match score"""
        query_terms = set(query.lower().split()) # Split query into terms, lowercase
        keyword_results = []

        for idx, meta in enumerate(self.metadata):
            text = meta.get('metadata', {}).get('text', '').lower() # Access text from nested metadata, lowercase
            if not text: # Handle cases where text might be missing or None
                continue

            score = sum(1 for term in query_terms if term in text) # Simple term frequency scoring
            if score > 0:
                result_meta = meta.copy() # Copy metadata to avoid modifying original
                result_meta['keyword_score'] = score # Add keyword score to metadata
                keyword_results.append(result_meta)

        # Sort by keyword score in descending order and limit to top_k
        return sorted(keyword_results, key=lambda x: x.get('keyword_score', 0), reverse=True)[:top_k]


    def _rerank_results(self, semantic_results: List[Dict], keyword_results: List[Dict], keyword_weight: float, top_k: int) -> List[Dict]:
        """Rerank combined results using Reciprocal Rank Fusion (RRF) - Combines semantic and keyword scores"""
        combined_results = {}
        rank = 1
        rrf_constant = 60 # Constant for RRF - adjust as needed

        # Process semantic results for RRF score
        for res in semantic_results:
            doc_id = res['chunk_checksum'] # Unique identifier for document chunk - using checksum
            if doc_id not in combined_results:
                combined_results[doc_id] = res.copy() # Initialize with semantic result
            combined_results[doc_id]['rrf_score_semantic'] = 1.0 / (rrf_constant + rank) # RRF score for semantic rank
            rank += 1

        # Process keyword results and fuse with semantic results
        rank = 1
        for res in keyword_results:
            doc_id = res['chunk_checksum'] # Unique identifier - using checksum
            if doc_id not in combined_results:
                combined_results[doc_id] = res.copy() # Initialize if not already in semantic results
                combined_results[doc_id]['rrf_score_semantic'] = 0.0 # Default semantic RRF score if only in keyword results

            combined_results[doc_id]['rrf_score_keyword'] = 1.0 / (rrf_constant + rank) # RRF score for keyword rank
            rank += 1


        # Calculate final RRF score - weighted combination of semantic and keyword RRF scores
        for doc_id in combined_results:
            semantic_rrf = combined_results[doc_id].get('rrf_score_semantic', 0.0)
            keyword_rrf = combined_results[doc_id].get('rrf_score_keyword', 0.0)
            combined_results[doc_id]['rrf_score_final'] = (1.0 - keyword_weight) * semantic_rrf + keyword_weight * keyword_rrf # Weighted RRF

        # Sort by final RRF score and get top_k
        ranked_results = sorted(combined_results.values(), key=lambda x: x.get('rrf_score_final', 0.0), reverse=True)[:top_k]
        return ranked_results


    def _format_results(self, results: List[Dict]) -> str:
        """Format search results for display - More informative output with metadata"""
        output_str = ""
        for i, res in enumerate(results):
            output_str += f"**Result {i+1}** (Score: RRF={res.get('rrf_score_final', 0):.2f}, Semantic={res.get('semantic_score', 0):.2f}, Keyword={res.get('keyword_score', 0):.2f}):\n" # More detailed scores
            output_str += f"- **Section:** {res['metadata'].get('section_type', 'N/A')}\n" # Section type
            output_str += f"- **Page(s):** {', '.join(map(str, res['metadata'].get('page_numbers', [])))}\n" # Page numbers
            output_str += f"- **Vulnerabilities:** {', '.join(res['metadata'].get('vulnerabilities', ['N/A']))}\n" # Vulnerabilities
            output_str += f"- **Text:**\n{res['text']}\n\n" # Chunk text

        return output_str

    def _recover_from_failure(self):
        """Self-healing mechanism - Resets storage in case of errors during add_documents"""
        logger.warning("Attempting to recover from failure by resetting vector store...")
        self._reset_storage() # Reset storage to clean state
        logger.info("Vector store reset. Ready for new documents.")
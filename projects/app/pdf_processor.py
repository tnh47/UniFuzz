import fitz
import re
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional # Import Optional
from app.config import Config
import datetime # Import datetime for timestamp


logger = logging.getLogger(__name__)

class PDFProcessor:
    SECTION_PATTERNS = {
        'vulnerability': re.compile(r'(CWE-\d+|\[(CRITICAL|HIGH|MEDIUM|LOW)\]|Vulnerability|Issue|Risk)', re.IGNORECASE), # More comprehensive vulnerability pattern
        'code_section': re.compile(r'```solidity(.*?)```', re.DOTALL),
        'recommendation': re.compile(r'(Recommendation|Mitigation|Fix):', re.IGNORECASE), # More recommendation keywords
        'severity': re.compile(r'Severity:\s*(\w+)', re.IGNORECASE), # Capture severity level
        'contract_name': re.compile(r'Contract:\s*(\w+)', re.IGNORECASE), # Example: Capture contract name if present
        'affected_function': re.compile(r'Function:\s*(\w+)', re.IGNORECASE) # Example: Capture affected function
        # Add more patterns as needed based on audit report formats
    }

    def process_pdf(self, pdf_path: Path) -> List[Dict]:
        """Enhanced PDF processing with layout analysis and metadata extraction"""
        try:
            with fitz.open(pdf_path) as doc:
                content = self._analyze_layout(doc) # Analyze layout first for better chunking
                full_text = "".join([item['text'] for item in content if item['type'] == 'text']) # Extract full text for overall metadata
                global_metadata = self._extract_global_metadata(full_text, pdf_path.name) # Extract metadata from full text
                chunks = self._hierarchical_chunking(content) # Hierarchical chunking based on layout
                enhanced_chunks = [self._enhance_metadata(c, global_metadata.copy()) for c in chunks] # Enhance each chunk with metadata
                return enhanced_chunks

        except Exception as e:
            logger.error(f"PDF processing failed for {pdf_path}: {str(e)}", exc_info=True) # Include exc_info for detailed traceback
            raise

    def _analyze_layout(self, doc) -> List[Dict]:
        """Analyze PDF layout to detect sections and code blocks - Returns structured content blocks"""
        content = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if b['type'] == 0:  # Text block
                    text = "\n".join([l['spans'][0]['text'] for l in b["lines"]])
                    content.append({
                        "text": text.strip(), # Strip whitespace for cleaner text
                        "type": "text",
                        "page": page_num + 1,
                        "bbox": b["bbox"]
                    })
                elif b['type'] == 1:  # Image block
                    content.append({
                        "type": "image",
                        "page": page_num + 1,
                        "bbox": b["bbox"]
                    })
        return content

    def _hierarchical_chunking(self, content: List[Dict]) -> List[Dict]:
        """Multi-level chunking strategy based on layout and sections - More structured chunking"""
        chunks = []
        current_chunk_text = "" # Accumulate text for current chunk
        current_metadata = {"page_numbers": [], "bboxes": []} # Metadata for current chunk
        current_section_type = None

        for item in content:
            if item['type'] == 'text':
                text_block = item['text']
                section_type = self._detect_section(text_block)

                if section_type and section_type != current_section_type: # New section starts
                    if current_chunk_text: # Save previous chunk if not empty
                        chunks.append(self._create_chunk(current_chunk_text, current_metadata, current_section_type))
                        current_chunk_text = "" # Reset chunk text
                        current_metadata = {"page_numbers": [], "bboxes": []} # Reset metadata

                    current_section_type = section_type # Update current section type

                current_chunk_text += text_block + "\n" # Add text block to current chunk
                current_metadata["page_numbers"].append(item['page']) # Track page numbers
                current_metadata["bboxes"].append(item['bbox']) # Track bounding boxes

                if len(current_chunk_text) > Config.CHUNK_SIZE: # Chunk size limit
                    chunks.append(self._create_chunk(current_chunk_text, current_metadata, current_section_type))
                    current_chunk_text = current_chunk_text[Config.CHUNK_OVERLAP:] # Overlap for next chunk - take last overlap chars
                    current_metadata = {"page_numbers": current_metadata["page_numbers"][-1:], "bboxes": current_metadata["bboxes"][-1:]} # Keep last page/bbox for overlap context


        if current_chunk_text: # Save the last chunk
            chunks.append(self._create_chunk(current_chunk_text, current_metadata, current_section_type))

        return chunks

    def _create_chunk(self, text: str, metadata: dict, section_type: str) -> Dict:
        """Create a chunk dictionary with text and metadata - Centralized chunk creation"""
        # Get min/max page numbers for the chunk
        page_numbers = sorted(list(set(metadata["page_numbers"]))) # Unique page numbers, sorted
        min_page = page_numbers[0]
        max_page = page_numbers[-1]

        chunk_metadata = {
            "page_numbers": page_numbers,
            "min_page": min_page,
            "max_page": max_page,
            "section_type": section_type,
            "text": text.strip(), # Store text in metadata for keyword search, strip whitespace
            "chunk_checksum": hashlib.md5(text.strip().encode()).hexdigest() # Calculate checksum here and store in metadata
        }
        return {"text": text.strip(), "metadata": chunk_metadata} # Separate text and metadata in the chunk dict

    def _detect_section(self, text: str) -> Optional[str]:
        """Detect section type using patterns - Returns section type string or None"""
        for section_type, pattern in self.SECTION_PATTERNS.items():
            if pattern.search(text):
                return section_type
        return None

    def _extract_global_metadata(self, full_text: str, pdf_filename: str) -> dict:
        """Extract metadata from the full PDF text - Example metadata extraction"""
        metadata = {
            "pdf_filename": pdf_filename,
            "report_checksum": hashlib.md5(full_text.encode()).hexdigest(), # Overall report checksum
            "extracted_at": str(datetime.datetime.now()) # Extraction timestamp
        }
        # Example: Extract contract name from the whole document if mentioned globally
        contract_match = self.SECTION_PATTERNS['contract_name'].search(full_text)
        if contract_match:
            metadata["contract_name"] = contract_match.group(1).strip()

        return metadata


    def _enhance_metadata(self, chunk: Dict, global_metadata: dict) -> Dict:
        """Add security-specific metadata to chunk - Inherits global metadata and adds chunk-specific info"""
        metadata = chunk["metadata"]
        metadata.update(global_metadata) # Inherit global metadata

        # Detect vulnerability info within the chunk
        vulnerabilities = self.SECTION_PATTERNS['vulnerability'].findall(chunk["text"])
        if vulnerabilities:
            metadata["vulnerabilities"] = list(set(vulnerabilities)) # Unique vulnerabilities in chunk

        severity_match = self.SECTION_PATTERNS['severity'].search(chunk["text"])
        if severity_match:
            metadata["severity"] = severity_match.group(1).strip() # Extract severity level

        affected_function_match = self.SECTION_PATTERNS['affected_function'].search(chunk["text"])
        if affected_function_match:
            metadata["affected_function"] = affected_function_match.group(1).strip() # Extract affected function

        return chunk

import datetime # Import datetime for timestamp
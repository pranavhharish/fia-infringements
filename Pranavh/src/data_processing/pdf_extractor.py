"""
PDF Text Extraction Module
Extracts text from FIA decision documents with robust error handling
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict
import PyPDF2
from tqdm import tqdm


class PDFExtractor:
    """Extract text from PDF documents"""

    def __init__(self, min_text_length: int = 100, max_text_length: int = 50000):
        """
        Initialize PDF extractor

        Args:
            min_text_length: Minimum valid text length
            max_text_length: Maximum text length to extract
        """
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text from a PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                # Validate extracted text
                if len(text) < self.min_text_length:
                    self.logger.warning(f"Text too short ({len(text)} chars) for {pdf_path.name}")
                    return None

                # Truncate if too long
                if len(text) > self.max_text_length:
                    text = text[:self.max_text_length]
                    self.logger.info(f"Truncated text to {self.max_text_length} chars for {pdf_path.name}")

                return text

        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path.name}: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,;:!?\-\'\"()\[\]{}]', '', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def extract_metadata_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract metadata from document text

        Args:
            text: Document text

        Returns:
            Dictionary with metadata fields
        """
        metadata = {
            'document_number': None,
            'date': None,
            'time': None,
            'session': None,
            'race': None
        }

        # Extract document number
        doc_match = re.search(r'Document\s*(\d+)', text, re.IGNORECASE)
        if doc_match:
            metadata['document_number'] = doc_match.group(1)

        # Extract date
        date_match = re.search(r'Date\s*(\d{1,2}\s+\w+\s+\d{4})', text, re.IGNORECASE)
        if date_match:
            metadata['date'] = date_match.group(1)

        # Extract time
        time_match = re.search(r'Time\s*(\d{1,2}:\d{2})', text, re.IGNORECASE)
        if time_match:
            metadata['time'] = time_match.group(1)

        # Extract session
        session_patterns = [
            r'Session\s*([A-Za-z\s]+?)(?:Fact|Offence|Decision)',
            r'(Practice|Qualifying|Race|Sprint)',
        ]
        for pattern in session_patterns:
            session_match = re.search(pattern, text, re.IGNORECASE)
            if session_match:
                metadata['session'] = session_match.group(1).strip()
                break

        # Extract race name from filename or text
        race_patterns = [
            r'([\w\s]+)\s+Grand Prix',
            r'([\w\s]+)\s+GRAND PRIX',
        ]
        for pattern in race_patterns:
            race_match = re.search(pattern, text)
            if race_match:
                metadata['race'] = race_match.group(1).strip()
                break

        return metadata

    def process_directory(self, directory: Path, output_dir: Path = None) -> Dict[str, Dict]:
        """
        Process all PDFs in a directory

        Args:
            directory: Directory containing PDFs
            output_dir: Optional directory to save text files

        Returns:
            Dictionary mapping filenames to extraction results
        """
        results = {}
        pdf_files = list(directory.glob("*.pdf"))

        self.logger.info(f"Processing {len(pdf_files)} PDF files from {directory}")

        for pdf_file in tqdm(pdf_files, desc=f"Processing {directory.name}"):
            text = self.extract_text_from_pdf(pdf_file)

            if text:
                cleaned_text = self.clean_text(text)
                metadata = self.extract_metadata_from_text(text)

                results[pdf_file.name] = {
                    'text': cleaned_text,
                    'raw_text': text,
                    'metadata': metadata,
                    'text_length': len(cleaned_text),
                    'success': True
                }

                # Save to text file if output directory provided
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    txt_file = output_dir / f"{pdf_file.stem}.txt"
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
            else:
                results[pdf_file.name] = {
                    'text': None,
                    'error': 'Extraction failed',
                    'success': False
                }

        success_count = sum(1 for r in results.values() if r['success'])
        self.logger.info(f"Successfully processed {success_count}/{len(pdf_files)} files")

        return results

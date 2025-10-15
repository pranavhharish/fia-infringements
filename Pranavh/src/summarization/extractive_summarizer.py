"""
Extractive Summarization Module
Uses LexRank and TextRank for extractive summarization
"""

import re
from typing import List
from collections import Counter
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
import logging


class ExtractiveSummarizer:
    """Extractive summarization using graph-based methods"""

    def __init__(self, method: str = 'lexrank', num_sentences: int = 5):
        """
        Initialize extractive summarizer

        Args:
            method: Summarization method ('lexrank', 'textrank', 'luhn')
            num_sentences: Number of sentences to extract
        """
        self.method = method.lower()
        self.num_sentences = num_sentences
        self.logger = logging.getLogger(__name__)

        # Initialize summarizer
        if self.method == 'lexrank':
            self.summarizer = LexRankSummarizer()
        elif self.method == 'textrank':
            self.summarizer = TextRankSummarizer()
        elif self.method == 'luhn':
            self.summarizer = LuhnSummarizer()
        else:
            raise ValueError(f"Unknown method: {method}")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for summarization

        Args:
            text: Raw text

        Returns:
            Preprocessed text
        """
        # Remove document headers and boilerplate
        text = re.sub(r'From The Stewards To.*?(?=Fact|Offence|Decision)', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def summarize(self, text: str) -> str:
        """
        Generate extractive summary

        Args:
            text: Document text

        Returns:
            Extractive summary
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)

            if not processed_text or len(processed_text) < 100:
                self.logger.warning("Text too short for summarization")
                return text[:200] if text else ""

            # Parse text
            parser = PlaintextParser.from_string(processed_text, Tokenizer("english"))

            # Generate summary
            summary_sentences = self.summarizer(parser.document, self.num_sentences)

            # Combine sentences
            summary = ' '.join([str(sentence) for sentence in summary_sentences])

            return summary

        except Exception as e:
            self.logger.error(f"Error in extractive summarization: {e}")
            return text[:300] if text else ""

    def summarize_batch(self, texts: List[str]) -> List[str]:
        """
        Summarize multiple texts

        Args:
            texts: List of texts

        Returns:
            List of summaries
        """
        summaries = []
        for text in texts:
            summary = self.summarize(text)
            summaries.append(summary)
        return summaries

    def extract_key_phrases(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract key phrases from text

        Args:
            text: Document text
            top_k: Number of key phrases to extract

        Returns:
            List of key phrases
        """
        # Simple key phrase extraction using word frequency
        text_lower = text.lower()

        # Remove stopwords (basic set)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'should', 'could', 'may', 'might', 'must', 'can', 'that',
                    'this', 'these', 'those', 'it', 'its', 'which', 'who', 'what', 'when',
                    'where', 'why', 'how'}

        # Extract bigrams and trigrams
        words = re.findall(r'\b[a-z]+\b', text_lower)

        # Bigrams
        bigrams = []
        for i in range(len(words) - 1):
            if words[i] not in stopwords and words[i+1] not in stopwords:
                bigrams.append(f"{words[i]} {words[i+1]}")

        # Trigrams
        trigrams = []
        for i in range(len(words) - 2):
            if (words[i] not in stopwords and
                words[i+1] not in stopwords and
                words[i+2] not in stopwords):
                trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        # Count frequencies
        phrase_freq = Counter(bigrams + trigrams)

        # Get top phrases
        top_phrases = [phrase for phrase, _ in phrase_freq.most_common(top_k)]

        return top_phrases

    def get_summary_stats(self, original_text: str, summary: str) -> dict:
        """
        Calculate summary statistics

        Args:
            original_text: Original text
            summary: Summary text

        Returns:
            Dictionary with statistics
        """
        stats = {
            'original_length': len(original_text),
            'summary_length': len(summary),
            'compression_ratio': len(summary) / len(original_text) if original_text else 0,
            'original_sentences': len(re.findall(r'[.!?]+', original_text)),
            'summary_sentences': len(re.findall(r'[.!?]+', summary))
        }
        return stats

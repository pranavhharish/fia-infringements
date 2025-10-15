"""
Abstractive Summarization Module
Uses transformer models (BART, T5, PEGASUS) for abstractive summarization
"""

import logging
from typing import List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)


class AbstractiveSummarizer:
    """Abstractive summarization using transformer models"""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        max_length: int = 150,
        min_length: int = 50,
        device: str = "cpu"
    ):
        """
        Initialize abstractive summarizer

        Args:
            model_name: HuggingFace model name
            max_length: Maximum summary length
            min_length: Minimum summary length
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Initialize model and tokenizer
        try:
            self.logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Move model to device
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            elif device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"

            # Create pipeline
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            self.logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def summarize(self, text: str, max_length: int = None, min_length: int = None) -> str:
        """
        Generate abstractive summary

        Args:
            text: Document text
            max_length: Override default max length
            min_length: Override default min length

        Returns:
            Abstractive summary
        """
        try:
            # Use default lengths if not specified
            max_len = max_length or self.max_length
            min_len = min_length or self.min_length

            # Truncate text if too long (most models have 1024 token limit)
            max_input_length = 1024
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)

            if len(inputs['input_ids'][0]) >= max_input_length:
                self.logger.warning(f"Input truncated to {max_input_length} tokens")

            # Generate summary
            summary = self.summarization_pipeline(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )

            return summary[0]['summary_text']

        except Exception as e:
            self.logger.error(f"Error in abstractive summarization: {e}")
            return text[:200] if text else ""

    def summarize_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Summarize multiple texts in batches

        Args:
            texts: List of texts
            batch_size: Batch size for processing

        Returns:
            List of summaries
        """
        summaries = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                batch_summaries = self.summarization_pipeline(
                    batch,
                    max_length=self.max_length,
                    min_length=self.min_length,
                    do_sample=False,
                    truncation=True
                )

                summaries.extend([s['summary_text'] for s in batch_summaries])

            except Exception as e:
                self.logger.error(f"Error in batch {i}: {e}")
                # Fallback: add truncated text
                summaries.extend([t[:200] for t in batch])

        return summaries

    def generate_team_summary(
        self,
        documents: List[str],
        team_name: str,
        year: str,
        max_summary_length: int = 200
    ) -> str:
        """
        Generate a team-year summary from multiple documents

        Args:
            documents: List of document texts
            team_name: Team name
            year: Year
            max_summary_length: Maximum summary length

        Returns:
            Team-year summary
        """
        # Concatenate all documents
        combined_text = " ".join(documents)

        # Create a context-aware prompt
        prompt = f"Summarize the key infringement patterns for {team_name} in {year}: {combined_text}"

        # Generate summary
        summary = self.summarize(prompt, max_length=max_summary_length, min_length=50)

        return summary

    def generate_insights(self, documents: List[str], num_insights: int = 5) -> List[str]:
        """
        Generate key insights from documents

        Args:
            documents: List of document texts
            num_insights: Number of insights to generate

        Returns:
            List of insights
        """
        insights = []

        # Combine documents
        combined_text = " ".join(documents)

        # Generate summary
        summary = self.summarize(combined_text, max_length=300)

        # Split summary into sentences
        sentences = summary.split('. ')

        # Take top N sentences as insights
        for i, sentence in enumerate(sentences[:num_insights]):
            if sentence and not sentence.endswith('.'):
                sentence += '.'
            insights.append(sentence.strip())

        return insights


class HybridSummarizer:
    """Combines extractive and abstractive summarization"""

    def __init__(
        self,
        extractive_summarizer,
        abstractive_summarizer
    ):
        """
        Initialize hybrid summarizer

        Args:
            extractive_summarizer: ExtractiveSummarizer instance
            abstractive_summarizer: AbstractiveSummarizer instance
        """
        self.extractive = extractive_summarizer
        self.abstractive = abstractive_summarizer
        self.logger = logging.getLogger(__name__)

    def summarize(self, text: str) -> dict:
        """
        Generate both extractive and abstractive summaries

        Args:
            text: Document text

        Returns:
            Dictionary with both summaries
        """
        return {
            'extractive': self.extractive.summarize(text),
            'abstractive': self.abstractive.summarize(text)
        }

    def summarize_with_fusion(self, text: str) -> str:
        """
        Generate summary using fusion approach:
        1. Extract key sentences using extractive method
        2. Refine using abstractive method

        Args:
            text: Document text

        Returns:
            Fused summary
        """
        # Step 1: Extract key content
        extractive_summary = self.extractive.summarize(text)

        # Step 2: Refine with abstractive summarization
        if len(extractive_summary) > 100:
            abstractive_summary = self.abstractive.summarize(extractive_summary)
            return abstractive_summary
        else:
            return extractive_summary

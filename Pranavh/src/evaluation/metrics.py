"""
Evaluation Metrics Module
Computes ROUGE, BLEU, and entity overlap metrics
"""

import logging
from typing import List, Dict
import numpy as np
from rouge_score import rouge_scorer
from collections import Counter
import re


class EvaluationMetrics:
    """Compute evaluation metrics for summarization"""

    def __init__(self):
        """Initialize evaluation metrics"""
        self.logger = logging.getLogger(__name__)
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def compute_rouge(self, summary: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE scores

        Args:
            summary: Generated summary
            reference: Reference summary

        Returns:
            Dictionary with ROUGE scores
        """
        try:
            scores = self.rouge_scorer.score(reference, summary)

            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge1_p': scores['rouge1'].precision,
                'rouge1_r': scores['rouge1'].recall,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rouge2_p': scores['rouge2'].precision,
                'rouge2_r': scores['rouge2'].recall,
                'rougeL_f': scores['rougeL'].fmeasure,
                'rougeL_p': scores['rougeL'].precision,
                'rougeL_r': scores['rougeL'].recall,
            }
        except Exception as e:
            self.logger.error(f"Error computing ROUGE: {e}")
            return {}

    def compute_bleu(self, summary: str, reference: str) -> float:
        """
        Compute BLEU score (simplified version)

        Args:
            summary: Generated summary
            reference: Reference summary

        Returns:
            BLEU score
        """
        try:
            # Tokenize
            summary_tokens = summary.lower().split()
            reference_tokens = reference.lower().split()

            # Compute n-gram precision for n=1,2,3,4
            precisions = []

            for n in range(1, 5):
                summary_ngrams = self._get_ngrams(summary_tokens, n)
                reference_ngrams = self._get_ngrams(reference_tokens, n)

                if len(summary_ngrams) == 0:
                    precision = 0
                else:
                    matches = sum((summary_ngrams & reference_ngrams).values())
                    precision = matches / sum(summary_ngrams.values())

                precisions.append(precision)

            # Compute geometric mean
            if all(p > 0 for p in precisions):
                bleu = np.exp(np.mean([np.log(p) for p in precisions]))
            else:
                bleu = 0

            # Brevity penalty
            bp = self._brevity_penalty(len(summary_tokens), len(reference_tokens))

            return bp * bleu

        except Exception as e:
            self.logger.error(f"Error computing BLEU: {e}")
            return 0.0

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)

    def _brevity_penalty(self, c: int, r: int) -> float:
        """Compute BLEU brevity penalty"""
        if c > r:
            return 1.0
        else:
            return np.exp(1 - r/c) if c > 0 else 0.0

    def compute_entity_overlap(
        self,
        summary: str,
        reference: str,
        entities: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute entity overlap between summary and reference

        Args:
            summary: Generated summary
            reference: Reference text
            entities: Optional list of key entities to check

        Returns:
            Dictionary with overlap metrics
        """
        if entities:
            # Check specific entities
            summary_lower = summary.lower()
            reference_lower = reference.lower()

            entities_in_summary = sum(1 for e in entities if e.lower() in summary_lower)
            entities_in_reference = sum(1 for e in entities if e.lower() in reference_lower)

            if entities_in_reference > 0:
                recall = entities_in_summary / entities_in_reference
            else:
                recall = 0.0

            if len(entities) > 0:
                precision = entities_in_summary / len(entities)
            else:
                precision = 0.0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                'entity_recall': recall,
                'entity_precision': precision,
                'entity_f1': f1,
                'entities_found': entities_in_summary,
                'total_entities': len(entities)
            }
        else:
            # Extract entities automatically (basic approach)
            summary_entities = self._extract_entities(summary)
            reference_entities = self._extract_entities(reference)

            common_entities = summary_entities & reference_entities

            if len(reference_entities) > 0:
                recall = len(common_entities) / len(reference_entities)
            else:
                recall = 0.0

            if len(summary_entities) > 0:
                precision = len(common_entities) / len(summary_entities)
            else:
                precision = 0.0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                'entity_recall': recall,
                'entity_precision': precision,
                'entity_f1': f1
            }

    def _extract_entities(self, text: str) -> set:
        """Extract entities (simple capitalized words)"""
        # Extract capitalized words as potential entities
        entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        return entities

    def compute_compression_ratio(self, summary: str, original: str) -> float:
        """
        Compute compression ratio

        Args:
            summary: Summary text
            original: Original text

        Returns:
            Compression ratio (0-1)
        """
        if len(original) == 0:
            return 0.0

        return len(summary) / len(original)

    def evaluate_summary(
        self,
        summary: str,
        reference: str = None,
        original: str = None,
        key_entities: List[str] = None
    ) -> Dict:
        """
        Comprehensive summary evaluation

        Args:
            summary: Generated summary
            reference: Reference summary (optional)
            original: Original text (optional)
            key_entities: Key entities to check (optional)

        Returns:
            Dictionary with all metrics
        """
        results = {
            'summary_length': len(summary),
            'summary_word_count': len(summary.split())
        }

        # ROUGE and BLEU (if reference provided)
        if reference:
            rouge_scores = self.compute_rouge(summary, reference)
            results.update(rouge_scores)

            bleu_score = self.compute_bleu(summary, reference)
            results['bleu'] = bleu_score

            # Entity overlap
            entity_scores = self.compute_entity_overlap(summary, reference, key_entities)
            results.update(entity_scores)

        # Compression ratio (if original provided)
        if original:
            results['compression_ratio'] = self.compute_compression_ratio(summary, original)

        return results

    def evaluate_batch(
        self,
        summaries: List[str],
        references: List[str] = None,
        originals: List[str] = None
    ) -> Dict:
        """
        Evaluate multiple summaries

        Args:
            summaries: List of generated summaries
            references: List of reference summaries (optional)
            originals: List of original texts (optional)

        Returns:
            Dictionary with average metrics
        """
        all_scores = []

        for i, summary in enumerate(summaries):
            reference = references[i] if references else None
            original = originals[i] if originals else None

            scores = self.evaluate_summary(summary, reference, original)
            all_scores.append(scores)

        # Compute averages
        avg_scores = {}
        if all_scores:
            for key in all_scores[0].keys():
                values = [s[key] for s in all_scores if key in s and s[key] is not None]
                if values:
                    avg_scores[f'avg_{key}'] = np.mean(values)
                    avg_scores[f'std_{key}'] = np.std(values)

        return avg_scores


class EntityExtractionEvaluator:
    """Evaluate entity extraction quality"""

    def __init__(self):
        """Initialize evaluator"""
        self.logger = logging.getLogger(__name__)

    def evaluate_extraction(
        self,
        extracted: Dict,
        ground_truth: Dict
    ) -> Dict[str, float]:
        """
        Evaluate entity extraction

        Args:
            extracted: Extracted entities
            ground_truth: Ground truth entities

        Returns:
            Precision, recall, F1 scores
        """
        results = {}

        # Evaluate each entity type
        for entity_type in ground_truth.keys():
            extracted_set = set(extracted.get(entity_type, []))
            truth_set = set(ground_truth[entity_type])

            if len(truth_set) == 0:
                precision = 1.0 if len(extracted_set) == 0 else 0.0
                recall = 1.0
            else:
                true_positives = len(extracted_set & truth_set)
                precision = true_positives / len(extracted_set) if len(extracted_set) > 0 else 0.0
                recall = true_positives / len(truth_set)

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            results[f'{entity_type}_precision'] = precision
            results[f'{entity_type}_recall'] = recall
            results[f'{entity_type}_f1'] = f1

        # Overall metrics
        precisions = [v for k, v in results.items() if k.endswith('_precision')]
        recalls = [v for k, v in results.items() if k.endswith('_recall')]
        f1s = [v for k, v in results.items() if k.endswith('_f1')]

        results['overall_precision'] = np.mean(precisions) if precisions else 0.0
        results['overall_recall'] = np.mean(recalls) if recalls else 0.0
        results['overall_f1'] = np.mean(f1s) if f1s else 0.0

        return results

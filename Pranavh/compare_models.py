#!/usr/bin/env python3
"""
Model Comparison Script
Compares multiple summarization methods with evaluation metrics and visualization
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import time
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_processing.pdf_extractor import PDFExtractor
from summarization.extractive_summarizer import ExtractiveSummarizer
from evaluation.metrics import EvaluationMetrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare multiple summarization models"""

    def __init__(self, sample_size: int = None):
        """
        Initialize model comparator

        Args:
            sample_size: Number of documents to test (None = all documents)
        """
        self.sample_size = sample_size
        self.pdf_extractor = PDFExtractor()
        self.evaluator = EvaluationMetrics()

        # Initialize summarizers
        self.models = {
            'LexRank': ExtractiveSummarizer(method='lexrank', num_sentences=5),
            'TextRank': ExtractiveSummarizer(method='textrank', num_sentences=5),
            'Luhn': ExtractiveSummarizer(method='luhn', num_sentences=5),
            'LexRank-3sent': ExtractiveSummarizer(method='lexrank', num_sentences=3),
            'TextRank-7sent': ExtractiveSummarizer(method='textrank', num_sentences=7)
        }

        self.results = []

    def load_sample_documents(self, base_path: Path) -> List[Dict]:
        """
        Load sample documents from the dataset

        Args:
            base_path: Base path to Documents folder

        Returns:
            List of document dictionaries
        """
        documents = []
        years = [
            '2020-infirdgement_profile',
            '2021-infridgement_profile',
            '2022-infridgement_profile',
            '2023-infridgement_profile',
            '2024-infridgement_profile'
        ]

        if self.sample_size:
            logger.info(f"Loading up to {self.sample_size} sample documents...")
            docs_per_year = self.sample_size // len(years)
        else:
            logger.info(f"Loading ALL documents from all years...")
            docs_per_year = None

        for year_folder in years:
            year_path = base_path / year_folder
            if not year_path.exists():
                continue

            pdf_files = list(year_path.glob("*.pdf"))
            if docs_per_year:
                pdf_files = pdf_files[:docs_per_year]

            for pdf_file in pdf_files:
                text = self.pdf_extractor.extract_text_from_pdf(pdf_file)
                if text:
                    cleaned_text = self.pdf_extractor.clean_text(text)
                    documents.append({
                        'filename': pdf_file.name,
                        'year': year_folder.split('-')[0],
                        'text': cleaned_text,
                        'word_count': len(cleaned_text.split())
                    })

                if self.sample_size and len(documents) >= self.sample_size:
                    break

            if self.sample_size and len(documents) >= self.sample_size:
                break

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def compare_models(self, documents: List[Dict]) -> pd.DataFrame:
        """
        Compare all models on the documents

        Args:
            documents: List of document dictionaries

        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing summarization models...")

        all_results = []

        for doc_idx, doc in enumerate(documents):
            logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {doc['filename']}")

            for model_name, model in self.models.items():
                start_time = time.time()

                # Generate summary
                try:
                    summary = model.summarize(doc['text'])
                    processing_time = time.time() - start_time

                    # Calculate metrics
                    summary_length = len(summary)
                    summary_word_count = len(summary.split())
                    compression_ratio = summary_length / len(doc['text']) if doc['text'] else 0

                    # Evaluate summary quality
                    eval_scores = self.evaluator.evaluate_summary(
                        summary=summary,
                        original=doc['text']
                    )

                    result = {
                        'document': doc['filename'],
                        'year': doc['year'],
                        'model': model_name,
                        'original_length': len(doc['text']),
                        'original_words': doc['word_count'],
                        'summary_length': summary_length,
                        'summary_words': summary_word_count,
                        'compression_ratio': compression_ratio,
                        'processing_time': processing_time,
                        'summary': summary[:200] + '...'  # First 200 chars
                    }

                    # Add evaluation scores
                    result.update(eval_scores)

                    all_results.append(result)

                except Exception as e:
                    logger.error(f"Error with {model_name} on {doc['filename']}: {e}")
                    continue

        return pd.DataFrame(all_results)

    def calculate_aggregate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregate metrics per model

        Args:
            df: Results dataframe

        Returns:
            Aggregate metrics dataframe
        """
        metrics = df.groupby('model').agg({
            'compression_ratio': ['mean', 'std'],
            'processing_time': ['mean', 'std'],
            'summary_words': ['mean', 'std'],
            'summary_length': 'mean'
        }).round(4)

        # Flatten column names
        metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
        metrics = metrics.reset_index()

        return metrics

    def visualize_results(self, df: pd.DataFrame, output_dir: Path):
        """
        Create visualization of model comparison

        Args:
            df: Results dataframe
            output_dir: Directory to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (14, 10)

        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Summarization Model Comparison', fontsize=16, fontweight='bold')

        # 1. Compression Ratio Comparison
        ax1 = axes[0, 0]
        df.boxplot(column='compression_ratio', by='model', ax=ax1)
        ax1.set_title('Compression Ratio by Model')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Compression Ratio')
        plt.sca(ax1)
        plt.xticks(rotation=45, ha='right')

        # 2. Processing Time Comparison
        ax2 = axes[0, 1]
        avg_time = df.groupby('model')['processing_time'].mean().sort_values()
        avg_time.plot(kind='barh', ax=ax2, color='steelblue')
        ax2.set_title('Average Processing Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Model')

        # 3. Summary Length Distribution
        ax3 = axes[0, 2]
        for model in df['model'].unique():
            model_data = df[df['model'] == model]['summary_words']
            ax3.hist(model_data, alpha=0.5, label=model, bins=15)
        ax3.set_title('Summary Length Distribution')
        ax3.set_xlabel('Words in Summary')
        ax3.set_ylabel('Frequency')
        ax3.legend()

        # 4. Compression Ratio vs Original Length
        ax4 = axes[1, 0]
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax4.scatter(model_data['original_words'],
                       model_data['compression_ratio'],
                       label=model, alpha=0.6)
        ax4.set_title('Compression Ratio vs Document Length')
        ax4.set_xlabel('Original Words')
        ax4.set_ylabel('Compression Ratio')
        ax4.legend()

        # 5. Average Metrics Heatmap
        ax5 = axes[1, 1]
        metrics_for_heatmap = df.groupby('model').agg({
            'compression_ratio': 'mean',
            'processing_time': 'mean',
            'summary_words': 'mean'
        })
        # Normalize for visualization
        metrics_normalized = (metrics_for_heatmap - metrics_for_heatmap.min()) / (metrics_for_heatmap.max() - metrics_for_heatmap.min())
        sns.heatmap(metrics_normalized.T, annot=True, fmt='.3f',
                   cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Normalized Score'})
        ax5.set_title('Normalized Metrics Heatmap')
        ax5.set_ylabel('Metric')

        # 6. Model Rankings
        ax6 = axes[1, 2]
        # Calculate overall score (lower compression = better, lower time = better)
        model_scores = df.groupby('model').agg({
            'compression_ratio': 'mean',
            'processing_time': 'mean',
            'summary_words': 'mean'
        })

        # Simple scoring: prefer moderate compression, fast processing
        model_scores['score'] = (
            (1 - model_scores['compression_ratio']) * 0.4 +  # Prefer more compression
            (1 / (1 + model_scores['processing_time'])) * 0.3 +  # Prefer faster
            (model_scores['summary_words'] / 100) * 0.3  # Prefer reasonable length
        )
        model_scores = model_scores.sort_values('score', ascending=False)

        model_scores['score'].plot(kind='barh', ax=ax6, color='green', alpha=0.7)
        ax6.set_title('Overall Model Score (Higher = Better)')
        ax6.set_xlabel('Score')
        ax6.set_ylabel('Model')

        plt.tight_layout()

        # Save plot
        plot_path = output_dir / 'model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {plot_path}")

        plt.close()

        # Create individual comparison plots
        self._create_detailed_plots(df, output_dir)

    def _create_detailed_plots(self, df: pd.DataFrame, output_dir: Path):
        """Create additional detailed plots"""

        # Plot 1: Time series comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        for model in df['model'].unique():
            model_data = df[df['model'] == model].sort_values('document')
            ax.plot(range(len(model_data)),
                   model_data['compression_ratio'].values,
                   marker='o', label=model, alpha=0.7)

        ax.set_title('Compression Ratio Across Documents')
        ax.set_xlabel('Document Index')
        ax.set_ylabel('Compression Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'compression_trend.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Performance comparison table
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')

        summary_stats = df.groupby('model').agg({
            'compression_ratio': ['mean', 'std', 'min', 'max'],
            'processing_time': ['mean', 'std'],
            'summary_words': ['mean', 'std']
        }).round(4)

        # Create table
        table_data = []
        for model in summary_stats.index:
            row = [
                model,
                f"{summary_stats.loc[model, ('compression_ratio', 'mean')]:.3f}",
                f"{summary_stats.loc[model, ('compression_ratio', 'std')]:.3f}",
                f"{summary_stats.loc[model, ('processing_time', 'mean')]:.3f}s",
                f"{summary_stats.loc[model, ('summary_words', 'mean')]:.0f}"
            ]
            table_data.append(row)

        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'Avg Compression', 'Std Compression', 'Avg Time', 'Avg Words'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.2, 0.2, 0.15, 0.15])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(output_dir / 'performance_table.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, df: pd.DataFrame, output_dir: Path):
        """
        Generate comprehensive comparison report

        Args:
            df: Results dataframe
            output_dir: Directory to save report
        """
        report_path = output_dir / 'comparison_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SUMMARIZATION MODEL COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")

            # Overview
            f.write(f"Total documents tested: {df['document'].nunique()}\n")
            f.write(f"Models compared: {len(df['model'].unique())}\n")
            f.write(f"Models: {', '.join(df['model'].unique())}\n\n")

            # Per-model statistics
            f.write("="*70 + "\n")
            f.write("MODEL PERFORMANCE METRICS\n")
            f.write("="*70 + "\n\n")

            for model in df['model'].unique():
                model_data = df[df['model'] == model]

                f.write(f"\n{model}:\n")
                f.write("-" * 50 + "\n")
                f.write(f"  Average Compression Ratio: {model_data['compression_ratio'].mean():.4f}\n")
                f.write(f"  Std Compression Ratio: {model_data['compression_ratio'].std():.4f}\n")
                f.write(f"  Average Processing Time: {model_data['processing_time'].mean():.4f}s\n")
                f.write(f"  Average Summary Length: {model_data['summary_words'].mean():.0f} words\n")
                f.write(f"  Min Compression: {model_data['compression_ratio'].min():.4f}\n")
                f.write(f"  Max Compression: {model_data['compression_ratio'].max():.4f}\n")

            # Rankings
            f.write("\n" + "="*70 + "\n")
            f.write("MODEL RANKINGS\n")
            f.write("="*70 + "\n\n")

            # Rank by compression ratio
            avg_compression = df.groupby('model')['compression_ratio'].mean().sort_values()
            f.write("Best Compression (Lower = Better):\n")
            for i, (model, ratio) in enumerate(avg_compression.items(), 1):
                f.write(f"  {i}. {model}: {ratio:.4f}\n")

            # Rank by speed
            avg_time = df.groupby('model')['processing_time'].mean().sort_values()
            f.write("\nFastest Processing:\n")
            for i, (model, time_val) in enumerate(avg_time.items(), 1):
                f.write(f"  {i}. {model}: {time_val:.4f}s\n")

            # Recommendations
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")

            best_compression = avg_compression.index[0]
            fastest = avg_time.index[0]

            f.write(f"🏆 Best Compression: {best_compression}\n")
            f.write(f"⚡ Fastest Processing: {fastest}\n")

            # Calculate balanced score
            model_scores = df.groupby('model').agg({
                'compression_ratio': 'mean',
                'processing_time': 'mean'
            })
            model_scores['balanced_score'] = (
                (1 - model_scores['compression_ratio']) * 0.6 +
                (1 / (1 + model_scores['processing_time'])) * 0.4
            )
            best_balanced = model_scores['balanced_score'].idxmax()

            f.write(f"⚖️  Best Balanced: {best_balanced}\n")

            f.write("\n" + "="*70 + "\n")

        logger.info(f"Saved report to {report_path}")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("F1 INFRINGEMENT ANALYSIS - MODEL COMPARISON")
    print("="*70 + "\n")

    # Initialize comparator (None = process all documents)
    comparator = ModelComparator(sample_size=None)

    # Load sample documents
    base_path = Path("../Documents")
    documents = comparator.load_sample_documents(base_path)

    if not documents:
        logger.error("No documents loaded. Please check the Documents path.")
        return

    # Compare models
    results_df = comparator.compare_models(documents)

    # Save detailed results
    output_dir = Path("outputs/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / 'detailed_results.csv', index=False)
    logger.info(f"Saved detailed results to {output_dir / 'detailed_results.csv'}")

    # Calculate aggregate metrics
    aggregate_metrics = comparator.calculate_aggregate_metrics(results_df)
    aggregate_metrics.to_csv(output_dir / 'aggregate_metrics.csv', index=False)
    logger.info(f"Saved aggregate metrics to {output_dir / 'aggregate_metrics.csv'}")

    # Generate visualizations
    comparator.visualize_results(results_df, output_dir)

    # Generate report
    comparator.generate_report(results_df, output_dir)

    # Print summary
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nDocuments tested: {len(documents)}")
    print(f"Models compared: {len(comparator.models)}")
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - detailed_results.csv")
    print("  - aggregate_metrics.csv")
    print("  - model_comparison.png")
    print("  - compression_trend.png")
    print("  - performance_table.png")
    print("  - comparison_report.txt")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

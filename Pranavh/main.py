#!/usr/bin/env python3
"""
F1 Infringement Analysis - Main Pipeline
Orchestrates the complete analysis workflow
"""

import argparse
import logging
import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_processing.pdf_extractor import PDFExtractor
from utils.team_detector import TeamDetector
from entity_extraction.entity_extractor import EntityExtractor
from summarization.extractive_summarizer import ExtractiveSummarizer
from summarization.abstractive_summarizer import AbstractiveSummarizer
from summarization.team_summarizer import TeamYearSummarizer
from evaluation.metrics import EvaluationMetrics


class F1InfringementAnalyzer:
    """Main analyzer for F1 infringement documents"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize analyzer

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_components()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))

        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Console handler
        if log_config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_config.get('file_output', True):
            log_file = Path(log_config.get('file', 'outputs/analysis.log'))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _initialize_components(self):
        """Initialize all processing components"""
        self.logger.info("Initializing components...")

        # Initialize extractors
        extraction_config = self.config.get('extraction', {})
        self.pdf_extractor = PDFExtractor(
            min_text_length=extraction_config.get('min_text_length', 100),
            max_text_length=extraction_config.get('max_text_length', 50000)
        )

        self.team_detector = TeamDetector('config/team_mappings.json')
        self.entity_extractor = EntityExtractor()

        # Initialize summarizers
        summarization_config = self.config.get('summarization', {})
        self.extractive_summarizer = ExtractiveSummarizer(
            method=summarization_config.get('extractive_method', 'lexrank'),
            num_sentences=summarization_config.get('extractive_sentences', 5)
        )

        # Note: Abstractive summarizer requires significant resources
        # Initialize only if needed
        self.abstractive_summarizer = None

        # Initialize team summarizer
        self.team_summarizer = TeamYearSummarizer(self.abstractive_summarizer)

        # Initialize evaluator
        if self.config.get('evaluation', {}).get('enabled', True):
            self.evaluator = EvaluationMetrics()
        else:
            self.evaluator = None

    def process_year_directory(self, year_dir: Path, year: str) -> List[Dict]:
        """
        Process all PDFs in a year directory

        Args:
            year_dir: Directory path
            year: Year string

        Returns:
            List of processed records
        """
        self.logger.info(f"Processing {year_dir.name}...")

        records = []
        pdf_files = list(year_dir.glob("*.pdf"))

        for pdf_file in tqdm(pdf_files, desc=f"Processing {year}"):
            # Extract text
            text = self.pdf_extractor.extract_text_from_pdf(pdf_file)

            if not text:
                continue

            # Clean text
            cleaned_text = self.pdf_extractor.clean_text(text)

            # Extract metadata
            metadata = self.pdf_extractor.extract_metadata_from_text(text)

            # Detect team
            team_result = self.team_detector.detect_team(text, year)

            if not team_result:
                self.logger.debug(f"Could not detect team for {pdf_file.name}")
                continue

            team_name, confidence = team_result

            # Get driver name if car number is available
            car_numbers = self.entity_extractor.extract_car_numbers(text)
            driver_name = None
            if car_numbers:
                driver_name = self.team_detector.get_driver_name(
                    team_name, car_numbers[0], year
                )

            # Create structured record
            record = self.entity_extractor.create_structured_record(
                filename=pdf_file.name,
                text=cleaned_text,
                team_name=team_name,
                year=year,
                driver_name=driver_name,
                metadata=metadata
            )

            # Add confidence score
            record['team_confidence'] = confidence

            # Generate extractive summary
            extractive_summary = self.extractive_summarizer.summarize(cleaned_text)
            record['extractive_summary'] = extractive_summary

            records.append(record)

        self.logger.info(f"Processed {len(records)} documents from {year}")
        return records

    def process_all_years(self) -> List[Dict]:
        """
        Process all year directories

        Returns:
            List of all records
        """
        all_records = []
        data_config = self.config.get('data', {})
        input_dir = Path(data_config.get('input_dir', 'Documents'))
        years = data_config.get('years', [])

        for year_folder in years:
            year_path = input_dir / year_folder
            if not year_path.exists():
                self.logger.warning(f"Directory not found: {year_path}")
                continue

            # Extract year from folder name
            year = year_folder.split('-')[0]

            # Process year
            year_records = self.process_year_directory(year_path, year)
            all_records.extend(year_records)

        return all_records

    def generate_team_summaries(self, records: List[Dict]) -> List[Dict]:
        """
        Generate team-year summaries

        Args:
            records: List of all records

        Returns:
            List of team-year summaries
        """
        self.logger.info("Generating team-year summaries...")

        summarization_config = self.config.get('summarization', {})
        team_year_config = summarization_config.get('team_year_summary', {})

        summaries = self.team_summarizer.generate_all_summaries(
            records,
            style=team_year_config.get('style', 'factual'),
            num_insights=team_year_config.get('num_insights', 5)
        )

        return summaries

    def export_results(self, records: List[Dict], summaries: List[Dict]):
        """
        Export results to various formats

        Args:
            records: List of all records
            summaries: List of team-year summaries
        """
        output_config = self.config.get('output', {})
        output_dir = Path(self.config.get('data', {}).get('output_dir', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export detailed records to CSV
        if output_config.get('save_csv', True):
            df = pd.DataFrame(records)
            csv_path = output_dir / 'infringement_records.csv'
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved records to {csv_path}")

        # Export team summaries to CSV
        if output_config.get('save_csv', True) and summaries:
            self.team_summarizer.export_summaries_to_csv(
                summaries,
                str(output_dir / 'team_year_summaries.csv')
            )

        # Export to JSON
        if output_config.get('save_json', True):
            json_path = output_dir / 'infringement_records.json'
            with open(json_path, 'w') as f:
                json.dump(records, f, indent=2)
            self.logger.info(f"Saved records to {json_path}")

            summaries_json_path = output_dir / 'team_year_summaries.json'
            with open(summaries_json_path, 'w') as f:
                json.dump(summaries, f, indent=2)
            self.logger.info(f"Saved summaries to {summaries_json_path}")

        # Export text summaries
        if output_config.get('save_txt', True) and summaries:
            txt_path = output_dir / 'team_summaries.txt'
            with open(txt_path, 'w') as f:
                for summary in summaries:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"{summary['team']} - {summary['year']}\n")
                    f.write(f"{'='*60}\n\n")
                    for insight in summary['insights']:
                        f.write(f"• {insight}\n")
                    f.write("\n")
            self.logger.info(f"Saved text summaries to {txt_path}")

    def run(self):
        """Run the complete analysis pipeline"""
        self.logger.info("Starting F1 Infringement Analysis...")

        # Process all documents
        records = self.process_all_years()
        self.logger.info(f"Total records processed: {len(records)}")

        if not records:
            self.logger.error("No records to process. Exiting.")
            return

        # Generate team summaries
        summaries = self.generate_team_summaries(records)
        self.logger.info(f"Generated {len(summaries)} team-year summaries")

        # Export results
        self.export_results(records, summaries)

        self.logger.info("Analysis complete!")

        # Print summary statistics
        self._print_summary_stats(records, summaries)

    def _print_summary_stats(self, records: List[Dict], summaries: List[Dict]):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)

        # Overall stats
        print(f"\nTotal documents processed: {len(records)}")

        # By year
        years = {}
        for record in records:
            year = record.get('year')
            years[year] = years.get(year, 0) + 1

        print("\nDocuments by year:")
        for year in sorted(years.keys()):
            print(f"  {year}: {years[year]}")

        # By team
        teams = {}
        for record in records:
            team = record.get('team')
            teams[team] = teams.get(team, 0) + 1

        print("\nDocuments by team:")
        for team in sorted(teams.keys()):
            print(f"  {team}: {teams[team]}")

        print(f"\nTeam-year summaries generated: {len(summaries)}")
        print("\n" + "="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='F1 Infringement Analysis Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    try:
        # Initialize and run analyzer
        analyzer = F1InfringementAnalyzer(args.config)

        if args.verbose:
            analyzer.logger.setLevel(logging.DEBUG)

        analyzer.run()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

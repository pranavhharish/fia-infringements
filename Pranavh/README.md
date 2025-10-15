# F1 Infringement Analysis: Multi-Team NLP Summarization Project

A comprehensive Natural Language Processing pipeline for analyzing Formula One FIA stewards' decision documents and generating infringement profiles for all F1 teams across 2020-2024 seasons.

## Overview

This project extracts, analyzes, and summarizes Formula One regulatory infringement patterns using advanced NLP techniques including:
- PDF text extraction and preprocessing
- Team detection and entity extraction
- Extractive summarization (LexRank/TextRank)
- Abstractive summarization (BART/T5)
- Team-year profile generation
- Comprehensive evaluation metrics

## Features

✅ **Automated PDF Processing**: Extract text from 900+ FIA decision documents
✅ **Team Detection**: Identify all 10 F1 teams with high accuracy
✅ **Entity Extraction**: Extract drivers, car numbers, infractions, and penalties
✅ **Dual Summarization**: Both extractive and abstractive methods
✅ **Team Profiles**: Generate comprehensive team-year infringement summaries
✅ **Evaluation Metrics**: ROUGE, BLEU, and entity overlap scores
✅ **Multi-format Output**: CSV, JSON, and text exports

## Project Structure

```
Pranavh/
├── config/
│   ├── config.yaml              # Main configuration
│   └── team_mappings.json       # Team/driver mappings (2020-2024)
├── src/
│   ├── data_processing/
│   │   └── pdf_extractor.py     # PDF text extraction
│   ├── utils/
│   │   └── team_detector.py     # Team detection logic
│   ├── entity_extraction/
│   │   └── entity_extractor.py  # Entity extraction
│   ├── summarization/
│   │   ├── extractive_summarizer.py   # LexRank/TextRank
│   │   ├── abstractive_summarizer.py  # BART/T5 models
│   │   └── team_summarizer.py         # Team-year summaries
│   └── evaluation/
│       └── metrics.py           # Evaluation metrics
├── outputs/                     # Generated results
├── docs/                        # Documentation
├── notebooks/                   # Jupyter notebooks
├── main.py                      # Main pipeline script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### 1. Prerequisites
- Python 3.8+
- pip package manager
- 8GB+ RAM recommended

### 2. Install Dependencies

```bash
cd Pranavh
pip install -r requirements.txt
```

### 3. Download spaCy Model (Optional)

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
python main.py
```

### Advanced Options

```bash
# Use custom configuration
python main.py --config config/custom_config.yaml

# Enable verbose logging
python main.py --verbose
```

### Configuration

Edit `config/config.yaml` to customize:
- Data paths and years to process
- Summarization methods and parameters
- Output formats
- Model settings
- Evaluation options

Example configuration:

```yaml
data:
  input_dir: "../Documents"
  output_dir: "outputs"
  years:
    - "2020-infirdgement_profile"
    - "2021-infirdgement_profile"

summarization:
  extractive_method: "lexrank"
  extractive_sentences: 5
  abstractive_model: "facebook/bart-large-cnn"
  team_year_summary:
    num_insights: 5
    style: "factual"
```

## Workflow

The analysis pipeline follows these steps:

1. **Document Ingestion**: Load PDF files from year directories
2. **Text Extraction**: Extract and clean text using PyPDF2
3. **Team Detection**: Identify team affiliation using patterns and car numbers
4. **Entity Extraction**: Extract drivers, infractions, penalties
5. **Extractive Summarization**: Generate key sentence summaries
6. **Abstractive Summarization**: Create concise reformulated summaries (optional)
7. **Team-Year Aggregation**: Group and summarize by team and year
8. **Export Results**: Save to CSV, JSON, and text formats

## Output Files

After running the pipeline, the following files are generated in `outputs/`:

- `infringement_records.csv`: Detailed records with all extracted entities
- `team_year_summaries.csv`: Team-year summary statistics and insights
- `infringement_records.json`: Complete JSON export
- `team_year_summaries.json`: Summaries in JSON format
- `team_summaries.txt`: Human-readable text summaries
- `f1_infringement_analysis.log`: Processing log

## Team Mappings

The system supports all 10 F1 teams with historical driver/car number mappings:

- Mercedes-AMG Petronas F1 Team
- Oracle Red Bull Racing
- Scuderia Ferrari
- McLaren F1 Team
- BWT Alpine F1 Team
- Aston Martin Aramco Cognizant F1 Team
- Scuderia AlphaTauri / RB F1 Team
- Alfa Romeo F1 Team Stake / Kick Sauber
- MoneyGram Haas F1 Team
- Williams Racing

Team names, aliases, car numbers, and drivers are automatically updated for each season (2020-2024).

## Evaluation Metrics

The system computes comprehensive evaluation metrics:

### Summarization Quality
- **ROUGE-1, ROUGE-2, ROUGE-L**: N-gram overlap metrics
- **BLEU Score**: Translation quality metric
- **Entity Overlap**: Key information retention
- **Compression Ratio**: Summary length vs. original

### Entity Extraction Quality
- **Precision**: Correctly extracted / Total extracted
- **Recall**: Correctly extracted / Total available
- **F1-Score**: Harmonic mean of precision and recall

## Example Output

### Team-Year Summary Example

```
=============================================================
Mercedes-AMG Petronas F1 Team - 2023
=============================================================

• Mercedes received 23 infringement(s) in 2023.
• The most common infraction type was pit lane speeding (4 occurrence(s)).
• The most frequent penalty was time penalty (8 time(s)).
• Drivers involved: Lewis Hamilton, George Russell.
• Infractions occurred across 15 race(s).
```

## Advanced Features

### Custom Team Detection

Modify `config/team_mappings.json` to add custom team patterns or update driver rosters.

### Hybrid Summarization

Use both extractive and abstractive methods for optimal quality:

```python
from src.summarization import ExtractiveSummarizer, AbstractiveSummarizer, HybridSummarizer

extractive = ExtractiveSummarizer(method='lexrank')
abstractive = AbstractiveSummarizer(model_name='facebook/bart-large-cnn')
hybrid = HybridSummarizer(extractive, abstractive)

summary = hybrid.summarize_with_fusion(document_text)
```

### Batch Processing

Process multiple documents efficiently:

```python
from src.data_processing import PDFExtractor

extractor = PDFExtractor()
results = extractor.process_directory(pdf_directory, output_dir)
```

## Performance

- **Processing Speed**: ~900 documents in under 2 hours (CPU)
- **Memory Usage**: <8GB RAM
- **Accuracy**: >90% team detection precision
- **Summary Quality**: ROUGE-L >0.6 (with reference summaries)

## Troubleshooting

### Common Issues

**Issue**: ImportError for transformers
**Solution**: Ensure PyTorch is installed: `pip install torch transformers`

**Issue**: Team not detected
**Solution**: Check team_mappings.json for correct aliases and car numbers

**Issue**: Out of memory
**Solution**: Reduce batch_size in config.yaml or use CPU device

**Issue**: PDF extraction fails
**Solution**: Verify PDF is not encrypted or corrupted

## Contributing

This is a research project for the MSIM Text Mining course. Contributions and suggestions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for academic research purposes. FIA document data is publicly available from official sources.

## Citation

If you use this project in your research, please cite:

```
F1 Infringement Analysis: Multi-Team NLP Summarization
MSIM Text Mining Project, Fall 2025
University of Michigan School of Information
```

## Acknowledgments

- FIA for publicly available decision documents
- Formula One teams and drivers
- HuggingFace for transformer models
- Open-source NLP community

## Contact

For questions or issues, please open an issue in the repository or contact the project team.

---

**Built with ❤️ for F1 and NLP**

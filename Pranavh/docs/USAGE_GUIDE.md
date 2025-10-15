# F1 Infringement Analysis - Usage Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Running the Pipeline](#running-the-pipeline)
5. [Understanding Outputs](#understanding-outputs)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Installation

### Step 1: Clone or Navigate to Project

```bash
cd Pranavh
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```python
python -c "import PyPDF2, pandas, transformers; print('All dependencies installed!')"
```

## Quick Start

### Run Complete Pipeline

```bash
python main.py
```

This will:
1. Process all PDF documents in `../Documents/`
2. Extract entities and metadata
3. Generate summaries
4. Export results to `outputs/`

### Expected Output

```
Starting F1 Infringement Analysis...
Processing 2020-infirdgement_profile...
Processing PDFs: 100%|██████████| 131/131
Processed 131 documents from 2020
...
Analysis complete!

ANALYSIS SUMMARY
================================================================
Total documents processed: 727
Documents by year:
  2020: 131
  2021: 175
  ...
```

## Configuration

### Basic Configuration

Edit `config/config.yaml`:

```yaml
data:
  input_dir: "../Documents"
  output_dir: "outputs"

summarization:
  extractive_method: "lexrank"  # or "textrank", "luhn"
  extractive_sentences: 5
```

### Team Mappings

Edit `config/team_mappings.json` to update team information:

```json
{
  "teams": {
    "Mercedes": {
      "official_name": "Mercedes-AMG Petronas F1 Team",
      "car_numbers": {
        "2024": [44, 63]
      },
      "drivers": {
        "2024": {
          "44": "Lewis Hamilton",
          "63": "George Russell"
        }
      }
    }
  }
}
```

## Running the Pipeline

### Full Pipeline

```bash
python main.py --config config/config.yaml
```

### Verbose Mode

```bash
python main.py --verbose
```

### Process Specific Years

Edit `config/config.yaml`:

```yaml
data:
  years:
    - "2023-infridgement_profile"
    - "2024-infridgement_profile"
```

## Understanding Outputs

### 1. Infringement Records (CSV)

`outputs/infringement_records.csv`

Columns:
- `filename`: PDF file name
- `year`: Year of infringement
- `team`: Team name
- `driver`: Driver name
- `car_numbers`: Car numbers involved
- `primary_infraction`: Main infraction type
- `penalty_type`: Type of penalty
- `penalty_description`: Penalty details
- `extractive_summary`: Generated summary

### 2. Team-Year Summaries (CSV)

`outputs/team_year_summaries.csv`

Columns:
- `team`: Team name
- `year`: Year
- `total_infractions`: Count of infractions
- `races_affected`: Number of races
- `drivers`: Drivers involved
- `summary`: Generated insights

### 3. JSON Exports

`outputs/infringement_records.json` - Complete structured data
`outputs/team_year_summaries.json` - Summaries in JSON format

### 4. Text Summaries

`outputs/team_summaries.txt` - Human-readable summaries

```
============================================================
Mercedes-AMG Petronas F1 Team - 2023
============================================================

• Mercedes received 23 infringement(s) in 2023.
• The most common infraction type was pit lane speeding...
```

## Advanced Usage

### Using Individual Components

#### PDF Extraction

```python
from src.data_processing import PDFExtractor

extractor = PDFExtractor()
text = extractor.extract_text_from_pdf(pdf_path)
cleaned_text = extractor.clean_text(text)
```

#### Team Detection

```python
from src.utils import TeamDetector

detector = TeamDetector('config/team_mappings.json')
team_name, confidence = detector.detect_team(text, year='2023')
```

#### Entity Extraction

```python
from src.entity_extraction import EntityExtractor

extractor = EntityExtractor()
entities = extractor.extract_all_entities(text, team_name)
```

#### Extractive Summarization

```python
from src.summarization import ExtractiveSummarizer

summarizer = ExtractiveSummarizer(method='lexrank', num_sentences=5)
summary = summarizer.summarize(text)
```

#### Abstractive Summarization

```python
from src.summarization import AbstractiveSummarizer

summarizer = AbstractiveSummarizer(
    model_name='facebook/bart-large-cnn',
    max_length=150
)
summary = summarizer.summarize(text)
```

#### Team-Year Summaries

```python
from src.summarization import TeamYearSummarizer

team_summarizer = TeamYearSummarizer()
summary = team_summarizer.generate_team_year_summary(
    records,
    team='Mercedes',
    year='2023',
    style='factual'
)
```

### Jupyter Notebook

Use the interactive notebook for exploration:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Batch Processing

Process multiple years in parallel:

```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def process_year(year_dir):
    # Processing logic
    pass

with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_year, year_directories)
```

### Custom Evaluation

```python
from src.evaluation import EvaluationMetrics

evaluator = EvaluationMetrics()
scores = evaluator.evaluate_summary(
    summary=generated_summary,
    reference=reference_summary,
    original=original_text
)
```

## Troubleshooting

### Issue: Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:/path/to/Pranavh/src"
```

### Issue: Team Not Detected

**Problem**: Team returns None

**Solution**:
1. Check `config/team_mappings.json` has correct team aliases
2. Verify year matches available mappings
3. Check PDF text extraction worked correctly

### Issue: Low Summary Quality

**Problem**: Summaries are poor quality

**Solution**:
1. Increase `extractive_sentences` in config
2. Try different summarization method (textrank vs lexrank)
3. For abstractive, try different model (T5, PEGASUS)

### Issue: Memory Error

**Problem**: Out of memory during processing

**Solution**:
1. Reduce `batch_size` in config
2. Set `device: "cpu"` in config
3. Process fewer documents at once
4. Increase system swap space

### Issue: PDF Extraction Fails

**Problem**: Cannot extract text from PDF

**Solution**:
1. Check PDF is not encrypted
2. Try alternative library: `pdfminer.six`
3. Manually verify PDF is valid
4. Check file permissions

### Issue: Slow Processing

**Problem**: Pipeline is very slow

**Solution**:
1. Enable parallel processing in config
2. Skip abstractive summarization (resource-intensive)
3. Use smaller summarization models
4. Process subset of years first

## Performance Optimization

### Speed Up Processing

1. **Use extractive only**: Disable abstractive summarization
2. **Parallel processing**: Set `max_workers: 8` in config
3. **Smaller models**: Use distilled BART models
4. **Cache results**: Enable resume mode

### Reduce Memory Usage

1. **Batch processing**: Process years sequentially
2. **Clear cache**: Delete transformers cache regularly
3. **Limit text length**: Set `max_text_length: 10000`
4. **Use CPU**: Avoid GPU for small batches

## Best Practices

1. **Always backup data** before running
2. **Test on single year** before full run
3. **Review configuration** carefully
4. **Check logs** for warnings/errors
5. **Validate outputs** with sample checks
6. **Version control** config changes

## Getting Help

1. Check logs: `outputs/f1_infringement_analysis.log`
2. Review configuration: `config/config.yaml`
3. Test components: Use Jupyter notebook
4. Enable verbose: `python main.py --verbose`
5. Check documentation: `README.md`

## Additional Resources

- **PRD**: `todo.md` - Product requirements
- **Code docs**: Inline documentation in source files
- **Examples**: `notebooks/exploratory_analysis.ipynb`
- **Config reference**: `config/config.yaml`

---

For more advanced usage and customization, refer to the source code documentation in the `src/` directory.

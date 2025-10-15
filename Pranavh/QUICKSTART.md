# F1 Infringement Analysis - Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install Dependencies (2 min)

```bash
cd Pranavh
pip install -r requirements.txt
```

## Step 2: Verify Data Location (30 sec)

Ensure your PDF documents are in the correct location:

```
Textmining_Project/
├── Documents/
│   ├── 2020-infirdgement_profile/  ← PDFs here
│   ├── 2021-infridgement_profile/  ← PDFs here
│   ├── 2022-infridgement_profile/  ← PDFs here
│   ├── 2023-infridgement_profile/  ← PDFs here
│   └── 2024-infridgement_profile/  ← PDFs here
└── Pranavh/                        ← You are here
```

## Step 3: Run the Analysis (2 min)

```bash
python main.py
```

That's it! The system will:
- ✅ Extract text from all PDFs
- ✅ Detect teams and extract entities
- ✅ Generate summaries
- ✅ Export results to `outputs/`

## View Results

### Quick Summary
```bash
cat outputs/team_summaries.txt
```

### Detailed Data
```bash
# Open CSV in Excel/Numbers
open outputs/team_year_summaries.csv

# Or view in terminal
head -20 outputs/team_year_summaries.csv
```

### Interactive Exploration
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Expected Output

```
Starting F1 Infringement Analysis...
Initializing components...
Processing 2020-infirdgement_profile...
Processing PDFs: 100%|████████████| 131/131
Processed 131 documents from 2020
...
Analysis complete!

ANALYSIS SUMMARY
================================================================
Total documents processed: 727
Team-year summaries generated: 48
================================================================
```

## Output Files

Your results will be in the `outputs/` folder:

- `infringement_records.csv` - All extracted data
- `team_year_summaries.csv` - Team summaries by year
- `team_summaries.txt` - Human-readable summaries
- `infringement_records.json` - Complete JSON export
- `f1_infringement_analysis.log` - Processing log

## Example Summary

```
============================================================
Mercedes-AMG Petronas F1 Team - 2023
============================================================

• Mercedes received 23 infringement(s) in 2023.
• The most common infraction type was pit lane speeding (4 occurrence(s)).
• The most frequent penalty was time penalty (8 time(s)).
• Drivers involved: Lewis Hamilton, George Russell.
• Infractions occurred across 15 race(s).
```

## Troubleshooting

### Issue: Dependencies fail to install
```bash
# Try upgrading pip first
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: "Documents not found"
```bash
# Check the path in config
cat config/config.yaml | grep input_dir
# Should show: input_dir: "../Documents"
```

### Issue: Slow processing
```bash
# Run on single year first to test
# Edit config/config.yaml and comment out years:
# years:
#   - "2023-infridgement_profile"  # Start with one year
```

### Issue: Import errors
```bash
# Ensure you're in the right directory
pwd  # Should end with /Pranavh

# Run from Pranavh folder
python main.py
```

## Next Steps

### 1. Customize Configuration
Edit `config/config.yaml` to:
- Change summarization methods
- Adjust number of sentences
- Enable/disable features
- Configure output formats

### 2. Explore Interactively
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### 3. Process Specific Teams
```python
from src.utils import TeamDetector

detector = TeamDetector('config/team_mappings.json')
# Your custom analysis here
```

### 4. Advanced Analysis
See `docs/USAGE_GUIDE.md` for:
- Custom entity extraction
- Advanced summarization options
- Evaluation metrics
- Batch processing

## Getting Help

- 📚 **Full Documentation**: `README.md`
- 📖 **Usage Guide**: `docs/USAGE_GUIDE.md`
- 📋 **Project Requirements**: `todo.md`
- 🐛 **Issues**: Check `outputs/f1_infringement_analysis.log`

## Tips

💡 **Start Small**: Test on one year first
💡 **Check Logs**: Review log file for errors
💡 **Use Jupyter**: Great for exploring results
💡 **Backup Config**: Save your config changes
💡 **Review Output**: Manually verify a few summaries

---

**Happy Analyzing! 🏎️**

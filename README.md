# Customer Insights Analysis

This repository contains a script that reads elder care community incident and weight data from an Excel workbook and produces community-level insights and CSV exports.

Files produced by the script

- `community_risk_analysis.csv` - combined community risk summary
- `incidents_detailed.csv` - flattened incident-level details
- `weights_detailed.csv` - flattened weight measurements
- `weight_changes_summary.csv` - per-resident weight change summary (if available)

Quick start

1. Create a Python environment (recommended):

   ```csh
   python3 -m venv venv
   source venv/bin/activate.csh
   ```

2. Install dependencies:

   ```csh
   python3 -m pip install --user -r requirements.txt
   ```

   Note: the `--user` flag installs packages to the current user's site-packages. If you activated the `venv` in step 1, omit `--user`.

3. Run the analysis:

   ```csh
   python3 customer-insights-analysis.py
   ```

4. Output CSV files will be written to the repository root.

Notes and recommendations

- The script expects the Excel file `Data Analyst Case - Dataset.xlsx` to be present in the repository root.
- Consider creating a `requirements.txt` (already included) and activating a virtual environment before running.
- Optional improvements: parameterize the input filename, add logging, and add unit tests for parsing utilities.

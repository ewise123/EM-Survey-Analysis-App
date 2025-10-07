# EM Skills Dashboard (Streamlit)

Interactive dashboard to explore Engagement Manager survey results. It auto-detects IMP (importance) and PERF (performance) questions, computes gaps (importance - performance), and provides visualizations, filters, and qualitative insights.

## Features
- Detects IMP/PERF columns from `EM Survey Results Cleaned.xlsx`
- Tidy long-format transformation
- Filters for demographic columns (any text column with small unique values)
- Skill gap ranking table with averages and medians
- Importance vs Performance scatter with diagonal reference
- Radar chart comparing Importance vs Performance across selected skills
- Gap distributions per skill
- Downloadable aggregate CSV
- Qualitative analysis: sentiment by question, top terms, and skill-linked insights
- Optional AI-powered qualitative summaries when you provide an OpenAI or Claude API key

## Project layout
- `app.py` - Streamlit app
- `requirements.txt` - Python dependencies
- Data expected at: `../EM Survey Results Cleaned.xlsx` (relative to this folder). You can also upload a file via the sidebar.

## Setup (Windows)
1. Install Python 3.11+ from https://www.python.org/downloads/
2. Open PowerShell in this folder: `em_dashboard/`
3. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
4. Install dependencies:
   - `pip install -r requirements.txt`
5. Run the app:
   - `streamlit run app.py`
6. Your browser will open to the dashboard. If not, visit the URL shown in the terminal.

## Using your own data
- Place the Excel file alongside the existing one and toggle "Use uploaded file" in the sidebar to upload a different workbook.
- The app looks for columns starting with `IMP` or `PERF` (case-insensitive), or containing words like `Importance`/`Performance`.
- Free-text responses belong on a sheet named `Qualitative`; the app reads it automatically when present.
- To generate AI-written summaries, expand "AI qualitative analysis (optional)" in the sidebar, choose a provider, and paste your API key (kept only in session memory). Install the matching SDK with `pip install openai` or `pip install anthropic`.

## Notes
- Scale is assumed to be 1-7. The app does not enforce range but charts default to that context.
- Positive `gap` indicates opportunity (importance exceeds performance). Negative indicates performance meets/exceeds importance.

## Troubleshooting
- If the app says it cannot detect IMP/PERF columns, check header names in your Excel or share a sample.
- For very wide screens, reduce zoom to see labels on scatter/radar clearly.
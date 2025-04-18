# Senate Stock Trading Analysis

This project tracks stock trading activities of US Senators, comparing their performance against the S&P 500 index and analyzing potential correlations between committee memberships and investment returns.

## Features

- Historical data collection of Senator stock trades
- Performance comparison against S&P 500
- Committee membership analysis
- Industry-specific return comparisons
- Interactive data visualization

## Project Structure

- `data/`: Raw and processed datasets
- `scripts/`: Data collection and processing scripts
- `analysis/`: Analysis notebooks and utilities
- `dashboard/`: Interactive visualization dashboard
- `tests/`: Test suite

## Setup

```bash
# Install required dependencies
pip install -r requirements.txt

# Run data collection
python scripts/collect_senate_data.py

# Launch dashboard
python dashboard/app.py
```

## Data Sources

- Senate trading data: [Senate Stock Watcher](https://senatestockwatcher.com/) and [Senate Stock Disclosure Portal](https://efdsearch.senate.gov/)
- S&P 500 data: Yahoo Finance API
- Committee membership: Official Senate records
- Industry returns: Various financial data providers # SI699FP

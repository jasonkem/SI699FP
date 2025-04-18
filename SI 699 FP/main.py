#!/usr/bin/env python3
"""
Main script to run the Senate stock trading analysis pipeline.
This script orchestrates data collection, analysis, and dashboard launch.
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_data_collection():
    """Run the data collection scripts."""
    logger.info("Starting data collection process...")
    
    # Run Senate trading data collection
    logger.info("Collecting Senate trading data...")
    try:
        subprocess.run([sys.executable, "scripts/collect_senate_data.py"], check=True)
        logger.info("Senate trading data collection completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Senate trading data collection failed: {e}")
        return False
    
    # Run S&P 500 and industry ETF data collection
    logger.info("Collecting S&P 500 and industry ETF data...")
    try:
        subprocess.run([sys.executable, "scripts/fetch_sp500_data.py"], check=True)
        logger.info("Market data collection completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Market data collection failed: {e}")
        return False
    
    logger.info("Data collection process completed")
    return True


def run_analysis():
    """Run the analysis scripts."""
    logger.info("Starting analysis process...")
    
    try:
        subprocess.run([sys.executable, "analysis/compare_performance.py"], check=True)
        logger.info("Performance comparison analysis completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Performance analysis failed: {e}")
        return False
    
    logger.info("Analysis process completed")
    return True


def launch_dashboard():
    """Launch the dashboard application."""
    logger.info("Launching dashboard...")
    
    try:
        # Using subprocess.Popen to run in the background
        process = subprocess.Popen(
            [sys.executable, "dashboard/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Check if process started successfully
        if process.poll() is None:
            logger.info("Dashboard started successfully (PID: %s)", process.pid)
            logger.info("Dashboard is accessible at http://localhost:8050")
            return True
        else:
            logger.error("Dashboard failed to start")
            return False
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Senate Stock Trading Analysis")
    
    parser.add_argument("--skip-collection", action="store_true", 
                        help="Skip data collection step")
    parser.add_argument("--skip-analysis", action="store_true", 
                        help="Skip analysis step")
    parser.add_argument("--skip-dashboard", action="store_true", 
                        help="Skip launching dashboard")
    
    return parser.parse_args()


def main():
    """Main function to run the analysis pipeline."""
    logger.info("Starting Senate stock trading analysis pipeline...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Ensure data directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("analysis/results", exist_ok=True)
    
    # Run data collection if not skipped
    if not args.skip_collection:
        if not run_data_collection():
            logger.error("Data collection failed. Aborting pipeline.")
            return 1
    else:
        logger.info("Skipping data collection as requested")
    
    # Run analysis if not skipped
    if not args.skip_analysis:
        if not run_analysis():
            logger.error("Analysis failed. Aborting pipeline.")
            return 1
    else:
        logger.info("Skipping analysis as requested")
    
    # Launch dashboard if not skipped
    if not args.skip_dashboard:
        if not launch_dashboard():
            logger.error("Dashboard launch failed.")
            return 1
    else:
        logger.info("Skipping dashboard launch as requested")
    
    logger.info("Senate stock trading analysis pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
#!/usr/bin/env python3
"""

This script serves as the main entry point for the sex-based 
classification pipeline. 

It handles:
    1. Data Preprocessing & Merging
    2. Exploratory Data Analysis (EDA) & Batch Effects Verification
    3. Dimensionality Reduction (PCA) & Baseline Classification (KNN)
    4. Advanced Predictive Modeling (XGBoost)

By executing this file, all components are run sequentially from scratch, ensuring
reproducibility and integration of all latest module updates.
"""

import subprocess
import sys
import time
from pathlib import Path


# Determine the absolute path to the root of the repository so that the pipeline
# can be executed from any working directory.
REPO_ROOT = Path(__file__).resolve().parent


# Each dictionary represents a sequential stage in the pipeline.
#   - "STEP1"       : Display step name for the console.
#   - "command"    : The exact shell command.
#   - "description": Explanation of what this stage achieves.
#   - "cwd"        : The working directory from which to run the command. 
#                    Defaults to REPO_ROOT. This handle Path name errors 
#                    and ensures that the script is run using the same Python 
#                    interpreter (and virtual environment) that executed run_pipeline.py.
PIPELINE_STEPS = [
    {
        "STEP1": "Data Preparation",
        # Using sys.executable ensures the script is run using the same Python 
        # interpreter (and virtual environment) that executed run_pipeline.py.
        "command": [sys.executable, "Data/dataprep.py"],
        "description": "Standardizes raw data inputs and merges disjoint metadata tables."
    },
    {
        "STEP2": "Exploratory Data Analysis",
        "command": [sys.executable, "Data/EDA+Batch_Effects.py"],
        "description": "Generates data distribution plots, cytokine correlation tables, and batch effect analysis."
    },
    {
        "STEP3": "PCA Notebook Conversion",
        # nbconvert dynamically converts the latest state of the IPython notebook
        # into a flat Python script, avoiding the need to manually sync them.
        "command": ["jupyter", "nbconvert", "--to", "script", "PCA_KNN.ipynb"],   
        # We specify cwd because the notebook has internal paths relative to the PCA+KNN directory.
        "cwd": REPO_ROOT / "PCA+KNN",
        "description": "Dynamically converts the latest PCA IPython notebook to a runnable Python script."
    },
    {
        "STEP4": "PCA & KNN Analysis",
        "command": [sys.executable, "PCA_KNN.py"],
        "cwd": REPO_ROOT / "PCA+KNN",
        "description": "Performs Principal Component Analysis (global and CV) and Baseline KNN Sex Classification."
    },
    {
        "STEP5": "XGBoost Evaluation",
        "command": [sys.executable, "XGBOOST/Xgboost.py"],
        "description": "Trains advanced predictive models, runs cross-validation, and assesses feature importances/AUROC."
    }
]


# HELPER FUNCTIONS

def print_banner(text):
    """Prints an organized banner to the console."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def run_step(step):
    """
    Executes a single pipeline step via the OS using subprocess.
    If the step raises an error, the entire pipeline halts to prevent cascading failures.
    """
    # Dynamically find the stage name whether the key is "name", "STEP1", "STEP2", etc.
    step_key = next((k for k in step.keys() if k == "name" or k.startswith("STEP")), "name")
    step_name = step.get(step_key, "Unknown Step")

    print_banner(f"STARTING STAGE: {step_name}")
    print(f"> {step['description']}")
    print(f"> Command: {' '.join(step['command'])}\n")
    
    start_time = time.time()
    
    # Extract the necessary working directory (defaults to the Repository Root)
    step_cwd = step.get("cwd", REPO_ROOT)
    
    # Run the subprocess synchronously. Standard output/error is printed continuously.
    result = subprocess.run(step["command"], cwd=step_cwd)
    
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    
    # Check if the subprocess threw an error (returncode != 0 denotes a failure)
    if result.returncode != 0:
        print_banner(f"FAILED STAGE: {step_name} (Runtime: {mins}m {secs}s)")
        print(f"Error executing: {' '.join(step['command'])}")
        # Exit the orchestrator forcefully, bubbling up the error code
        sys.exit(result.returncode)
        
    print_banner(f"FINISHED STAGE: {step_name} (Runtime: {mins}m {secs}s)")

#Main execution 
def main():
    print_banner("END-TO-END PIPELINE CLASSIFICATION PIPELINE RUN")
    
    total_start = time.time()
    
    # Execute each pipeline step sequentially
    for step in PIPELINE_STEPS:
        run_step(step)
        
    total_elapsed = time.time() - total_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)
    
    # Total time taken for the entire pipeline to run successfully across all components.
    # Final success banner displaying completion and output locations
    print("\n" + "=" * 80)
    print(f" PIPELINE COMPLETED SUCCESSFULLY (Total Runtime: {total_mins}m {total_secs}s)")
    print("=" * 80)
    print("✓ All data generated cleanly without errors.")
    print("✓ Standardized datasets saved to:    Data/")
    print("✓ Analysis & Plots successfully exported to the internal 'Outputs/' folders")
    print("  inside the 'PCA+KNN/' and 'XGBOOST/' directories.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

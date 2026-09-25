import os
import shutil
import subprocess
import sys

# Set Kaggle config dir to current directory so it picks up kaggle.json if placed here
os.environ["KAGGLE_CONFIG_DIR"] = os.path.dirname(os.path.abspath(__file__))

commands = [
    ("datasets", "sgpjesus/bank-account-fraud-dataset-neurips-2022", "06_projects_finance/02_fraud_detection"),
    ("competitions", "ieee-fraud-detection", "06_projects_finance/02_fraud_detection"),
    ("datasets", "saurabhbadole/leading-indian-bank-and-cibil-real-world-dataset", "06_projects_finance/03_credit_scoring"),
    ("datasets", "prakharrathi25/banking-dataset-marketing-targets", "06_projects_finance/04_marketing_propensity"),
    ("datasets", "ksabishek/massive-bank-dataset-1-million-rows", "06_projects_finance/05_massive_bank_data"),
]

use_uv = shutil.which("uv") is not None
runner = ["uv", "run", "kaggle"] if use_uv else [sys.executable, "-m", "kaggle"]

failed_downloads = []

for dl_type, name, path in commands:
    print(f"Downloading {name} to {path}...")

    # Ensure path exists
    os.makedirs(path, exist_ok=True)

    if dl_type == "competitions":
        cmd = runner + [
            "competitions", "download",
            "-c", name,
            "-p", path,
            "--unzip",
        ]
    elif dl_type == "datasets":
        cmd = runner + [
            "datasets", "download",
            "-d", name,
            "-p", path,
            "--unzip",
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode == 0:
        print(f"SUCCESS: {name}")
    else:
        print(f"FAILED: {name}")
        print(result.stderr)
        failed_downloads.append(name)
        if "403" in result.stderr or "Forbidden" in result.stderr:
            print("Note: If this is a competition, you must accept the rules on the Kaggle website first!")
    print("-" * 40)

if failed_downloads:
    print(f"\nCompleted with {len(failed_downloads)} failure(s): {', '.join(failed_downloads)}")
    sys.exit(1)

print("\nAll datasets downloaded successfully!")

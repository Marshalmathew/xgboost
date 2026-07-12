import os
import subprocess
import sys

# Set Kaggle config dir to current directory so it picks up kaggle.json
os.environ["KAGGLE_CONFIG_DIR"] = os.path.dirname(os.path.abspath(__file__))

commands = [
    ("datasets", "sgpjesus/bank-account-fraud-dataset-neurips-2022", r"06_projects_finance\02_fraud_detection"),
    ("competitions", "ieee-fraud-detection", r"06_projects_finance\02_fraud_detection"),
    ("datasets", "saurabhbadole/leading-indian-bank-and-cibil-real-world-dataset", r"06_projects_finance\03_credit_scoring"),
    ("datasets", "prakharrathi25/banking-dataset-marketing-targets", r"06_projects_finance\04_marketing_propensity"),
    ("datasets", "ksabishek/massive-bank-dataset-1-million-rows", r"06_projects_finance\05_massive_bank_data")
]

for dl_type, name, path in commands:
    print(f"Downloading {name} to {path}...")
    
    # Ensure path exists
    os.makedirs(path, exist_ok=True)
    
    cmd = [
        "uv", "run", "kaggle", dl_type, "download", 
        name, 
        "-p", path, 
        "--unzip"
    ]
    
    # If competition, use `-c` flag implicitly handled by `competitions download -c`
    # Wait, the command is `kaggle competitions download -c name`. But `kaggle competitions download name` also works in newer versions. Let's be safe.
    if dl_type == "competitions":
        cmd = [
            "uv", "run", "kaggle", "competitions", "download", 
            "-c", name, 
            "-p", path, 
            "--unzip"
        ]
    elif dl_type == "datasets":
        cmd = [
            "uv", "run", "kaggle", "datasets", "download", 
            "-d", name, 
            "-p", path, 
            "--unzip"
        ]
        
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"SUCCESS: {name}")
    else:
        print(f"FAILED: {name}")
        print(result.stderr)
        if "403" in result.stderr or "Forbidden" in result.stderr:
            print("Note: If this is a competition, you must accept the rules on the Kaggle website first!")
    print("-" * 40)

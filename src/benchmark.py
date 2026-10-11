import yaml
import pandas as pd

def run_benchmark():
    print("Initializing test experiment pipeline...")
    
    # Load configuration
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"Experiment: {config['experiment_name']}")
    print(f"Using random_seed: {config['random_seed']}")
    
    # Load dataset dynamically from config parameters
    data_path = config.get("data_params", {}).get("raw_data_path", "data/sample/survey.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded rows of data: {len(df)}")
    
    # Extract target column name from config
    target_col = config.get("data_params", {}).get("target_col", "Last Academic Result (GPA/CGPA)")
    
    # Convert GPA column to numeric and calculate average
    df["gpa_numeric"] = pd.to_numeric(df[target_col], errors="coerce")
    avg_gpa = df["gpa_numeric"].mean()
    
    print(f"Average GPA in dataset: {avg_gpa:.2f}")
    print("Pipeline executed successfully!")

if __name__ == "__main__":
    run_benchmark()

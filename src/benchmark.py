import yaml
import pandas as pd

def run_benchmark():
    print("Initializing test experiment pipeline...")
    
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Using random_seed: {config['random_seed']}")
    
    df = pd.read_csv("data/sample/sample_data.csv")
    print(f"Loaded rows of data: {len(df)}")
    
    avg_gpa = df["gpa"].mean()
    print(f"Average GPA in the test sample: {avg_gpa:.2f}")
    print("Pipeline executed successfully!")

if __name__ == "__main__":
    run_benchmark()

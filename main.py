# src/main.py
import pandas as pd
from rulefit import RuleFit
from preprocess import load_data, handle_missing_values, preprocess_data

def main():
    # Step 1: Load and preprocess data
    data_path = "../data/Germancreditcardfraud.csv"
    data = load_data(data_path)
    
    # Step 2: Handle missing values
    data = handle_missing_values(data)
    
    # Step 3: Extract features and target
    target_column = 'Class'
    X, y = preprocess_data(data, target_column)
    
    # Step 4: Train the RuleFit model
    rf = RuleFit(tree_size=4, max_rules=2000, rfmode='classify', model_type="rl", random_state=1, max_iter=1000)
    rf.fit(X, y, feature_names=X.columns)
    
    # Step 5: Extract and display rules
    rules = rf.get_rules()
    print("Extracted Rules:")
    print(rules.sort_values(by=["support"], ascending=False))

if __name__ == "__main__":
    main()

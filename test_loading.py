#!/usr/bin/env python3
"""
Test script for heat stress analysis
"""
import json
import pandas as pd

def test_data_loading():
    print("Testing data loading...")
    
    # Load JSON data
    with open('heat_stress_poultry.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check year data
    print(f"Year values sample: {df['year'].head(10).tolist()}")
    print(f"Year data type: {df['year'].dtype}")
    
    # Check abstracts
    print(f"Non-null abstracts: {df['abstract'].notna().sum()}")
    print(f"Empty abstracts: {(df['abstract'] == '').sum()}")
    
if __name__ == "__main__":
    test_data_loading()

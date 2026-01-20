import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

def main():
    csv_file = "crypto_data/top_250_crypto.csv"
    print("reading...")
    df = pd.read_csv(csv_file)
    print(f"Success loaded {len(df)} rows and {len(df.columns)} columns")

    print()
    print("first five rows:")
    print(df.head())

    print()
if __name__ == "__main__":
    main()

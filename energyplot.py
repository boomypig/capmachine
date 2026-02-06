import numpy as np #type:ignore
import pandas as pd #type:ignore

CSV_PATH="renewable_energy/EnergyproductionDataset.csv"

SEED = 119

# TARGET_COLUMN=
df = pd.read_csv(CSV_PATH)

print(df.tail())

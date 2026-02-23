import matplotlib as plot
import pandas as pd
import numpy as np
def main():
    df = pd.read_csv("./data/spotify_songs.csv")
    print(df.head)
    print(df.columns)
main()
import os
import numpy as np
import pandas as pd

paths: list[str] = []

for dirname, _, filenames in os.walk(os.getcwd()):
    for filename in filenames:
        if filename.endswith('.xlsx'):
            paths.append(os.path.join(dirname, filename))

df: pd.DataFrame = pd.read_excel(paths[0], index_col=0)

if __name__ == "__main__":
    print(df.shape)

import pandas as pd
# import matplotlib.pyplot as plt

a = pd.read_csv("abs01.txt", delim_whitespace=True)


print(a.iloc[0])

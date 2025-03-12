import pandas as pd

def Denoising(df, width=7):
    index = df.index
    columns = df.columns
    arrayin = df.values.copy()
    arrayout = arrayin.copy()
    for i in range(width, arrayin.shape[0] - width):
        temp = 0
        for j in range(i-width, i+width+1):
            temp = temp + arrayin[j]
        arrayout[i] = temp / (2*width+1)
    return pd.DataFrame(arrayout, columns=columns, index=index)
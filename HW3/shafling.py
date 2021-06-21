import pandas as pd
data = pd.read_csv('dataset/car.csv')
data = data.sample(frac=1)
data.to_csv('dataset/s_car.csv',index=False) 
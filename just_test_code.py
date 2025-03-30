import pandas as pd
import pickle

with open('df.pkl','rb') as file:
    df = pickle.load(file)
with open('models/model.pkl','rb') as file:
    model = pickle.load(file)
# print(df.columns)

# print(model)

df=pd.read_csv('data/processed/imputed_data.csv')

print(df.dtypes)
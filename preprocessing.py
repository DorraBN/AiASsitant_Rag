import pandas as pd
import numpy as np
def load_data():
    df=pd.read_csv("dataset.csv")
    print(df.head())
    return df
df=load_data()
def prepare_data(df):
    cat=df.select_dtypes(include="object").columns.tolist()
    num=df.select_dtypes(include="number").columns.tolist()
    for col in cat :
      df[col]=df[col].str.lower()
      df[col]=df[col].str.replace(r'[^\w\s]','',regex=True)
    return df

def miss_data(df):
   duplicated=df.duplicated().sum()
   nullv=df.isnull().sum()
   return duplicated,nullv

def prepare_documents(df):
    documents = []

    for _, row in df.iterrows():
        document = f"""
        Category: {row['category']}
        Product: {row['product']}
        Question: {row['question']}
        Answer: {row['answer']}
        """

        documents.append(document.strip())
    return documents
import streamlit as st
from preprocessing import load_data,prepare_data,miss_data,prepare_documents
st.set_page_config(page_title="AI assistant",layout="wide")
st.title("AI ASISTANT")
@st.cache_data
def get_data():
    df=load_data()
    return df
data=get_data()
st.write(data.head())
data_cleaned=prepare_data(data)
dupl,miss=miss_data(data)
st.write("Missing values per column:")
st.write(miss)
st.write("duplicated ",dupl)
st.write("data after cleaning")
st.write(data_cleaned.head())

doc=prepare_documents(data_cleaned)
st.write("Documents")
st.write(doc)

response=st.Page("pages/response.py"
)


import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from category_encoders import BinaryEncoder
from catboost import CatBoostRegressor

st.set_page_config(layout='wide', page_title='Flat Steel Prices')

html_title = """<h1 style="color:white;text-align:center;"> Flat Steel Prices </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

st.image('https://www.shutterstock.com/image-photo/packed-rolls-steel-sheet-cold-600nw-338337974.jpg')

df = pd.read_csv('cleaned_df.csv', index_col= 0)
st.dataframe(df.head())

Loaded_line = st.selectbox('Loaded_line', df.Loaded_line.unique())
KS_GRADE01 = st.selectbox('KS_GRADE01', df.KS_GRADE01.unique())
CUST_THICKNESS = st.sidebar.slider('CUST_THICKNESS', min_value=0.2, max_value=2.25, step=0.05)
CUST_WIDTH = st.sidebar.slider('CUST_WIDTH', min_value=265, max_value=1300, step=5)
ZINC01 = st.sidebar.slider('ZINC01', min_value=0, max_value=330, step=10)
Destination = st.sidebar.selectbox('Destination', df.Destination.unique())
Sector = st.sidebar.selectbox('Sector', df.Sector.unique())
Creation = st.selectbox('Creation', df.Creation.unique())
Material = st.selectbox('Material', df.Material.unique())

ml_model = joblib.load('catboost.pkl')

if st.button('Predict Flat Steel Price'):

    new_data = pd.DataFrame(columns= df.columns.drop('CBE$'), data= [[Loaded_line, Destination, Material, CUST_THICKNESS,
       CUST_WIDTH, KS_GRADE01, ZINC01, Sector, Creation]])

    st.write('Flat Steel Price :', ml_model.predict(new_data).round(2)[0])


import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from category_encoders import BinaryEncoder
from catboost import CatBoostRegressor

st.set_page_config(layout='wide', page_title=' 💰 Flat Steel Prices 💰 ')

st.header("🏭 Flat Steel Prices 🏭 ")
st.markdown("---") 

html_title = """<h1 style="color:white;text-align:center;"> Flat Steel Prices </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

st.image('https://www.shutterstock.com/image-photo/packed-rolls-steel-sheet-cold-600nw-338337974.jpg', 
         caption='لفائف الصلب المعدة للشحن')

df = pd.read_csv('cleaned_df.csv', index_col= 0)
st.dataframe(df.head())
st.subheader("📊 نظرة سريعة على البيانات التاريخية")

st.sidebar.title("⚙️ مواصفات المنتج")

st.sidebar.subheader("📐 الأبعاد والطلاء")
CUST_THICKNESS = st.sidebar.slider('السُمك (Thickness)', min_value=0.2, max_value=2.25, step=0.05)
CUST_WIDTH = st.sidebar.slider('العرض (Width)', min_value=265, max_value=1300, step=5)
ZINC01 = st.sidebar.slider('كمية الزنك (ZINC01)', min_value=0, max_value=330, step=10)

st.sidebar.subheader("🌍 تفاصيل الصفقة")
Destination = st.sidebar.selectbox('الوجهة (Destination)', df.Destination.unique())
Sector = st.sidebar.selectbox('القطاع (Sector)', df.Sector.unique())


st.subheader("اختيار المكونات الفئوية (Categorical Features)")
# تقسيم إلى 4 أعمدة لتنظيم المدخلات
col_A, col_B, col_C, col_D = st.columns(4)

with col_A:
    Loaded_line = st.selectbox('خط التحميل (Loaded Line)', df.Loaded_line.unique())
with col_B:
    KS_GRADE01 = st.selectbox('درجة المادة (KS Grade)', df.KS_GRADE01.unique())
with col_C:
    Creation = st.selectbox('نوع الإنشاء (Creation)', df.Creation.unique())
with col_D:
    Material = st.selectbox('المادة (Material)', df.Material.unique())


ml_model = joblib.load('catboost.pkl')

if st.button('Predict Flat Steel Price'):

    new_data = pd.DataFrame(columns= df.columns.drop('CBE$'), data= [[Loaded_line, Destination, Material, CUST_THICKNESS,
       CUST_WIDTH, KS_GRADE01, ZINC01, Sector, Creation]])

    st.write('Flat Steel Price :', ml_model.predict(new_data).round(2)[0])

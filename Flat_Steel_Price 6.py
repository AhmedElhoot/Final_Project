
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from category_encoders import BinaryEncoder
from catboost import CatBoostRegressor


st.set_page_config(layout='wide', page_title='🏭 Flat Steel Prices ')

st.header("🏭 Flat Steel Prices ")

html_title = """<h1 style="color:Black;text-align:center;font-size:65px;"> Flat Steel Prices </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

st.image('https://www.shutterstock.com/image-photo/packed-rolls-steel-sheet-cold-600nw-338337974.jpg', 
         caption=' Flat Steel Coils '
         use_column_width=False, 
             width=450))
st.markdown("---")

df = pd.read_csv('cleaned_df.csv', index_col= 0)
st.dataframe(df.head())
st.subheader("📊   Some Historical Data  ")

page=st.sidebar.radio('Pages',['Welcome Page','Univariate Numerical or Categorical Histograms','Univariate Numerical Box Plots','Univariate Categorical Pie Charts','Multivariete Analysis Bar Charts'])
if page == 'Univariate Numerical or Categorical Histograms':
    st.title('Univariate Numerical or Categorical Histograms')  
    for col in df.columns:
        st.plotly_chart(px.histogram(data_frame=df,x=col,title=col))
elif page == 'Univariate Numerical Box Plots':
    st.title('Univariate Numerical Box Plots')
    st.plotly_chart(px.box(data_frame=df,x='ZINC01', width=800, height=400))
    st.plotly_chart(px.box(data_frame=df,x='CBE$', width=800, height=400))
    st.plotly_chart(px.box(data_frame=df,x='BM$', width=800, height=400))
elif page == 'Univariate Categorical Pie Charts':
    st.title('Univariate Categorical Pie Charts')
    st.plotly_chart(px.pie(data_frame=df,names='Loaded_line',width=800, height=500))
    st.plotly_chart(px.pie(data_frame=df,names='Creation',width=800, height=500))
    st.plotly_chart(px.pie(data_frame=df,names='Destination', width=800, height=500))
    st.plotly_chart(px.pie(data_frame=df,names='Sector', width=800, height=500))
elif page == 'Multivariete Analysis Bar Charts':
    st.title('Multivariete Analysis Bar Charts')
    Remain_GRF_per_Demanded_Month_Per_Plan_Month=(df.groupby(['Demanded_Month','Plan_Month'])['Remain_GRF']).mean().round(2)
    df_plot = Remain_GRF_per_Demanded_Month_Per_Plan_Month.reset_index(name='Remain_GRF')
    st.plotly_chart(px.bar(
    data_frame=df_plot,
    x='Demanded_Month',
    y='Remain_GRF',
    color='Plan_Month',
    text_auto=True,
    title='Remain_GRF per Demanded Month per Plan Month',
    barmode='group'))
    Remain_GRF_per_Destination_Per_Loaded_line=(df.groupby(['Destination','Loaded_line'])['Remain_GRF']).mean().round(2)
    df_plot2 = Remain_GRF_per_Destination_Per_Loaded_line.reset_index(name='Remain_GRF')
    st.plotly_chart(px.bar(
    data_frame=df_plot2,
    x='Destination',
    y='Remain_GRF',
    color='Loaded_line',
    text_auto=True,
    title='Remain_GRF_per_Destination_Per_Loaded_line',
    barmode='group'))

st.sidebar.title("⚙️ Product Specifications ")


st.sidebar.subheader("📐 Size and Coating ")
CUST_THICKNESS = st.sidebar.slider('(Thickness)', min_value=0.2, max_value=2.25, step=0.05)
CUST_WIDTH = st.sidebar.slider('(Width)', min_value=265, max_value=1300, step=5)
ZINC01 = st.sidebar.slider('(Zinc Weight)', min_value=0, max_value=330, step=10)

st.sidebar.subheader("🌍 Where to deliver? ")
Destination = st.sidebar.selectbox(' (Destination)', df.Destination.unique())
Sector = st.sidebar.selectbox(' (Sector)', df.Sector.unique())


st.subheader("(Categorical Features)")
col_A, col_B, col_C, col_D = st.columns(4)
with col_A:
    Loaded_line = st.selectbox('(Loaded Line)', df.Loaded_line.unique())
with col_B:
    KS_GRADE01 = st.selectbox('(KS Grade)', df.KS_GRADE01.unique())
with col_C:
    Creation = st.selectbox('(Creation)', df.Creation.unique())
with col_D:
    Material = st.selectbox('(Material)', df.Material.unique())


ml_model = joblib.load('catboost.pkl')

if st.button('🚀 Predict Now', use_container_width=True, type='primary'):
    
    try:
        new_data = pd.DataFrame(
            columns= df.columns.drop('CBE$'), 
            data= [[
                Loaded_line, Destination, Material, CUST_THICKNESS,
                CUST_WIDTH, KS_GRADE01, ZINC01, Sector, Creation
            ]]
        )

        with st.spinner(' Predicting Now . . . '):
            prediction_value = ml_model.predict(new_data).round(2)[0]
        
        st.subheader("🎯 Predicted Price ")
        
        st.metric(
            label=" Price in CBE $", 
            value=f"{prediction_value:,.2f} $",   
            delta="Prediced By CatBoost Model",
            delta_color="normal"
        )
        
    except Exception as e:
        st.error(f"Error: {e}. Plz Revise Faulty Inputs.")

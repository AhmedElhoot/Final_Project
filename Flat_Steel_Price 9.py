
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from category_encoders import BinaryEncoder
from catboost import CatBoostRegressor
import plotly.express as px


st.set_page_config(layout='wide', page_title='🏭 Flat Steel Prices ')

st.header("🏭 Welcome to Flat Steel Price Predictor . . . ")
st.markdown("---")

html_title = """<h1 style="color:Black;font-size:65px;"> Flat Steel Prices </h1>"""
st.markdown(html_title, unsafe_allow_html=True)
st.markdown("---")
df = pd.read_csv('cleaned_df.csv', index_col= 0)

page=st.sidebar.radio('Pages',['Welcome Page','Company Data Analysis','Price Predictor For Your Order'])

if page == 'Welcome Page':
    st.markdown('<div style="text-align:center;"><img src="https://media1.tenor.com/m/GQ14lnCIGnsAAAAd/gerdau-work.gif" width="1000"><p><em>Universe of Steel</em></p></div>', unsafe_allow_html=True)
    st.subheader("📊   Some Historical Data  ")
    st.dataframe(df.head(15))


elif page == 'Company Data Analysis':    
    st.title('🏷️ Categorical Histograms')
    for col in df.select_dtypes('object').columns:
        st.plotly_chart(px.histogram(df,x=col,title=col,category_orders={col:df[col].value_counts().index.tolist()},
            color=col,color_discrete_sequence=px.colors.sequential.Agsunset,opacity=0.7,width=800,height=400,
            template='plotly_white').update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'))

    st.title('📊 Numerical Bar Charts')
    for col in df.select_dtypes(['int64','float64']).columns:
        c=df[col].value_counts(bins=10).sort_values(ascending=False)
        st.plotly_chart(px.bar(x=c.index.astype(str),y=c.values,title=col,color=c.values,
            color_continuous_scale=px.colors.sequential.Tealgrn,opacity=0.6,width=800,height=400,
            template='plotly_white').update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'))

    for x,color in [('ZINC01','#FF7F50'),('CBE$','#20B2AA')]:
        st.plotly_chart(px.box(df,x=x,title=f'{x} Boxplot',color_discrete_sequence=[color],width=800,height=400)
            .update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'))

    st.title('🥧 Categorical Pie Charts')
    for col in ['Loaded_line','Creation','Destination','Sector']:
        st.plotly_chart(px.pie(df,names=col,title=f'{col} Orders %',
            color_discrete_sequence=px.colors.sequential.Plasma_r,hole=0.3,width=800,height=500)
            .update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'))


elif page == 'Price Predictor For Your Order':
    st.markdown('<style>.stApp{background:linear-gradient(rgba(255,255,255,0.9),rgba(255,255,255,0.9)), url("https://img.etimg.com/thumb/width-420,height-315,imgsize-88152,resizemode-75,msid-120400173/news/economy/foreign-trade/duty-imposed-on-steel-aluminium-on-security-grounds-not-safeguard-measures-us-to-india-in-wto/bokaro-steel-plant-gets-rs-20k-crore-expansion-plan.jpg") center/cover no-repeat fixed; font-family:"Segoe UI",Tahoma,Verdana,sans-serif;}</style>', unsafe_allow_html=True)
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

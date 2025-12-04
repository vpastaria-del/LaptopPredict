import streamlit as st
import pickle
import numpy as np
import pandas as pd  # <-- required

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

# UI inputs
company = st.selectbox('Brand', df['Company'].unique())
type_ = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No','Yes'])
ips = st.selectbox('IPS', ['No','Yes'])
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160','3200x1800',
     '2880x1800','2560x1600','2560x1440','2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):

    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # compute ppi
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    # build DataFrame with EXACT columns the model expects
    input_df = pd.DataFrame([{
        'Company': company,
        'TypeName': type_,
        'Cpu brand': cpu,
        'Gpu brand': gpu,
        'os': os,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'ppi': ppi,
        'HDD': hdd,
        'SSD': ssd
    }])

    # prediction
    log_price = pipe.predict(input_df)[0]
    price = np.exp(log_price)
    usd_rate = 90
    price_in_usd = price / usd_rate
    st.subheader(f"Predicted price: $ {price_in_usd:,.2f}")
  
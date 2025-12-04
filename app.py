import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model and dataframe
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor with SHAP Explainability")

# Extract components from pipeline
preprocessor = pipe.named_steps['pre']     # ColumnTransformer
model = pipe.named_steps['model']          # RandomForestRegressor

# ---- UI Inputs ----
company = st.selectbox('Brand', df['Company'].unique())
type_ = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
weight = st.number_input('Laptop Weight')
touchscreen = st.selectbox('Touchscreen', ['No','Yes'])
ips = st.selectbox('IPS Display', ['No','Yes'])
screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160','3200x1800',
     '2880x1800','2560x1600','2560x1440','2304x1440']
)
cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())


if st.button("Predict Price"):

    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    # Calculate PPI
    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    # Create input dataframe
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

    # Prediction (your model predicts log-price)
    log_price = pipe.predict(input_df)[0]
    price = np.exp(log_price)
    price_usd = price / 90
    st.subheader(f"Predicted Price: $ {price_usd:,.2f}")

    # -----------------------------------------
    # -------- SHAP EXPLAINABILITY ------------
    # -----------------------------------------
    st.subheader("🔍 Why This Price Was Predicted (SHAP Explanation)")

    # 1) Transform input using preprocessor
    transformed_input = preprocessor.transform(input_df)

    # 2) Convert sparse -> dense -> float
    if not isinstance(transformed_input, np.ndarray):
        transformed_input = transformed_input.toarray()

    transformed_input = transformed_input.astype(float)

    # 3) Build correct feature names after OHE + scaling
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out(
        ['Ram','Weight','Touchscreen','Ips','ppi','HDD','SSD']
    ).tolist()

    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(
        ['Company','TypeName','Cpu brand','Gpu brand','os']
    ).tolist()

    feature_names = num_features + cat_features

    # 4) SHAP for RandomForest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_input)

    # ----- Local SHAP Explanation (Waterfall Plot) -----
    st.write("🔵 Negative impact | 🔴 Positive impact")

    fig, ax = plt.subplots(figsize=(10,6))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[0],   # FIX: must be scalar
        shap_values[0],                # SHAP values for this input
        feature_names=feature_names
    )
    st.pyplot(fig)

    # ----- Global Importance -----
    st.subheader("📊 Global Feature Importance")

    fig2, ax2 = plt.subplots(figsize=(10,6))
    shap.summary_plot(
        shap_values,
        transformed_input,
        feature_names=feature_names,
        show=False
    )
    st.pyplot(fig2)

import streamlit as st
import pandas as pd
import joblib
model = joblib.load("model/logistic_regression.pkl")
columns = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PaperlessBilling',
    'MonthlyCharges', 'TotalCharges',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'Contract_One year', 'Contract_Two year',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
]

scaled = joblib.load("model/scaler.pkl")
label_encoder_dependents = joblib.load("model/label_encoder_dependents.pkl")
label_encoder_partner = joblib.load("model/label_encoder_partner.pkl")
label_encoder_gender = joblib.load("model/label_encoder_gender.pkl")
label_encoder_phone_service = joblib.load("model/label_encoder_phoneservice.pkl")
label_encoder_paperless_billing = joblib.load("model/label_encoder_paperlessbilling.pkl")
label_encoder_churn = joblib.load("model/label_encoder_churn.pkl")

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()


    df = df.drop(columns=["customerID", "Churn"], errors="ignore")

    
    df["gender"]           = label_encoder_gender.transform(df["gender"])
    df["Partner"]          = label_encoder_partner.transform(df["Partner"])
    df["Dependents"]       = label_encoder_dependents.transform(df["Dependents"])
    df["PhoneService"]     = label_encoder_phone_service.transform(df["PhoneService"])
    df["PaperlessBilling"] = label_encoder_paperless_billing.transform(df["PaperlessBilling"])



    # TotalCharges -> numérique
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    
    cat_cols = ["PaymentMethod", "Contract", "StreamingMovies", "InternetService",
                "MultipleLines", "OnlineSecurity", "DeviceProtection",
                "StreamingTV", "TechSupport", "OnlineBackup"]
    df = pd.get_dummies(df, columns=cat_cols)

    # Garder seulement les 26 colonnes du modèle (gender, PhoneService,
    # MultipleLines_* sont éliminées ici), dans le bon ordre
    df = df.reindex(columns=columns, fill_value=0)

    # Normalisation
    return pd.DataFrame(scaled.transform(df), columns=columns, index=df.index)

st.set_page_config(page_title="customer_churn", page_icon="https://img.icons8.com/?size=100&id=f354PeNeLLyN&format=png&color=000000")
st.title("Customer Churn Prediction")
st.write("This is a simple web application that predicts customer churn based on user input features. Please provide the necessary information below to get started.")
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx","xls","csv"])
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
    st.dataframe(df)
    n= st.number_input(label="selected row", min_value=0, max_value=len(df), value=0, step=1)
    st.write(df.iloc[n])
    if st.button("Predict", key="predict_button"):
        X = preprocess(df)
        pred  = model.predict(X.iloc[[n]])[0]
        label = label_encoder_churn.inverse_transform([pred])[0]
        proba = model.predict_proba(X.iloc[[n]])[0][1]

        st.write(f"The predicted churn for the selected customer is: **{label}** ({proba:.1%})")
    
    
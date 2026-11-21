import streamlit as st
import pandas as pd
import joblib
model = joblib.load("model/logistic_regression.pkl")






st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


def load_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        css = f.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )
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


load_css("style/style.css")
st.title("Customer Churn Prediction")
st.write("This is a simple web application that predicts customer churn based on user input features. Please provide the necessary information below to get started.")

YES_NO_INTERNET = ["Yes", "No", "No internet service"]

with st.form("churn_form"):
    st.subheader("Informations personnelles")
    c1, c2, c3 ,c4 = st.columns(4)
    senior     = c1.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner    = c2.selectbox("Partner", ["Yes", "No"])
    dependents = c3.selectbox("Dependents", ["Yes", "No"])
    gender     = c4.selectbox("Gender", ["Male", "Female"])

    st.subheader("Contrat et facturation")
    c1, c2 = st.columns(2)
    tenure   = c1.number_input("Tenure (mois)", min_value=0, max_value=100, value=12, step=1)
    contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    c1, c2 = st.columns(2)
    paperless = c1.selectbox("Paperless billing", ["Yes", "No"])
    payment   = c2.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])

    c1, c2 = st.columns(2)
    monthly = c1.number_input("Monthly charges", min_value=0.0, value=70.0, step=0.5)
    total   = c2.number_input("Total charges", min_value=0.0, value=840.0, step=1.0)

    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    c1, c2, c3 = st.columns(3)
    security   = c1.selectbox("Online security", YES_NO_INTERNET)
    backup     = c2.selectbox("Online backup", YES_NO_INTERNET)
    protection = c3.selectbox("Device protection", YES_NO_INTERNET)
    c1, c2, c3 = st.columns(3)
    support = c1.selectbox("Tech support", YES_NO_INTERNET)
    tv      = c2.selectbox("Streaming TV", YES_NO_INTERNET)
    movies  = c3.selectbox("Streaming movies", YES_NO_INTERNET)

    submitted = st.form_submit_button("Predict")

if submitted:
    data = pd.DataFrame([{
        "gender": gender,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PaperlessBilling": paperless,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "PaymentMethod": payment,
        "Contract": contract,
        "StreamingMovies": movies,
        "InternetService": internet,
        "OnlineSecurity": security,
        "DeviceProtection": protection,
        "StreamingTV": tv,
        "TechSupport": support,
        "OnlineBackup": backup,
    }])

    X = preprocess(data)
    pred  = model.predict(X)[0]
    label = label_encoder_churn.inverse_transform([pred])[0]
    proba = model.predict_proba(X)[0][1]

    if pred == 1:
        st.error(f"Churn prédit : **{label}** (probabilité {proba:.1%})")
    else:
        st.success(f"Churn prédit : **{label}** (probabilité {proba:.1%})")
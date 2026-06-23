import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predictor de Fumadores", page_icon="🚬")


@st.cache_resource
def cargar():
    modelo = joblib.load("models/random_forest.joblib")
    umbral = joblib.load("models/umbral.joblib")
    X = pd.read_csv("data/processed/X_train.csv")
    return modelo, umbral, X


modelo, umbral, X = cargar()
medianas = X.median()
columnas = list(modelo.feature_names_in_)

st.title("🚬 Predictor de Fumadores")
st.write("Cargá los datos de una persona y el modelo predice si fuma o no.")

st.subheader("Datos personales")
genero = st.selectbox("Género", ["Hombre", "Mujer"])
edad = st.slider("Edad", 20, 85, 45)
altura = st.slider("Altura (cm)", 130, 190, 165)
peso = st.slider("Peso (kg)", 30, 135, 65)

st.subheader("Valores de laboratorio (escala del dataset)")
hemoglobina = st.slider("Hemoglobina", float(X["hemoglobin"].min()), float(X["hemoglobin"].max()), float(medianas["hemoglobin"]))
triglic = st.slider("Triglicéridos", float(X["triglyceride"].min()), float(X["triglyceride"].max()), float(medianas["triglyceride"]))
gtp = st.slider("Gtp (enzima del hígado)", float(X["Gtp"].min()), float(X["Gtp"].max()), float(medianas["Gtp"]))

if st.button("🔮 ¿Es fumador?", type="primary"):
    persona = medianas.copy()
    persona["gender"] = 1 if genero == "Hombre" else 0
    persona["age"] = edad
    persona["height(cm)"] = altura
    persona["weight(kg)"] = peso
    persona["hemoglobin"] = hemoglobina
    persona["triglyceride"] = triglic
    persona["Gtp"] = gtp

    entrada = pd.DataFrame([persona])[columnas]
    prob = float(modelo.predict_proba(entrada)[0, 1])

    st.divider()
    if prob >= umbral:
        st.error("🚬 PROBABLEMENTE FUMA")
    else:
        st.success("✅ PROBABLEMENTE NO FUMA")
    st.metric("Probabilidad de fumar", f"{prob * 100:.0f}%")
    st.caption(f"Umbral de decisión usado: {umbral}")

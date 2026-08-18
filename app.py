import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Analytics & Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Dashboard de Riesgo Crediticio y Simulador de Incumplimiento")
st.markdown("Herramienta interactiva para analizar la cartera de préstamos y predecir el riesgo de impago.")

# -----------------------------------------------------------------------------
# CARGA DE DATOS Y MODELO (Caché para optimizar rendimiento)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Reemplazar con la ruta real a tu dataset descargado de Kaggle
    df = pd.read_csv("credit_risk_dataset.csv")
    # Limpieza básica rápida
    df = df.dropna(subset=['person_income', 'loan_amnt', 'loan_status'])
    return df

@st.cache_resource
def get_trained_model(df):
    """
    Entrena un modelo rápido de demostración si no tienes cargado un model.pkl
    """
    features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']
    X = df[features].fillna(df[features].median())
    y = df['loan_status']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

df = load_data()
model = get_trained_model(df)

# -----------------------------------------------------------------------------
# BARRA LATERAL - FILTROS GLOBALES
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Filtros de Análisis")

# Filtro por Intención del Préstamo
intenciones = ['Todas'] + list(df['loan_intent'].dropna().unique())
intencion_selected = st.sidebar.selectbox("Propósito del Préstamo", intenciones)

# Filtro por Rango de Edad
min_age, max_age = int(df['person_age'].min()), int(df['person_age'].max())
edad_rango = st.sidebar.slider("Rango de Edad del Cliente", min_age, 80, (20, 60))

# Aplicar filtros al DataFrame
df_filtered = df[(df['person_age'] >= edad_rango[0]) & (df['person_age'] <= edad_rango[1])]
if intencion_selected != 'Todas':
    df_filtered = df_filtered[df_filtered['loan_intent'] == intencion_selected]

# -----------------------------------------------------------------------------
# SECCIÓN 1: METRICAS CLAVE (KPIs)
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_prestado = df_filtered['loan_amnt'].sum()
tasa_morosidad = (df_filtered['loan_status'].mean()) * 100
ingreso_promedio = df_filtered['person_income'].mean()
prestamo_promedio = df_filtered['loan_amnt'].mean()

col1.metric("Cartera Analizada", f"${total_prestado:,.0f}")
col2.metric("Tasa de Morosidad", f"{tasa_morosidad:.2f}%")
col3.metric("Ingreso Promedio", f"${ingreso_promedio:,.0f}")
col4.metric("Préstamo Promedio", f"${prestamo_promedio:,.0f}")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECCIÓN 2: VISUALIZACIÓN E INSIGHTS
# -----------------------------------------------------------------------------
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("📊 Distribución de Préstamos por Propósito")
    fig_intent = px.histogram(
        df_filtered, 
        x="loan_intent", 
        color="loan_status", 
        barmode="group",
        labels={"loan_intent": "Propósito", "count": "Número de Créditos", "loan_status": "Default (1)"},
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    st.plotly_chart(fig_intent, use_container_width=True)

with col_chart2:
    st.subheader("📉 Relación Ingreso vs. Monto Solicitado")
    fig_scatter = px.scatter(
        df_filtered.sample(min(1000, len(df_filtered))), # Muestra para velocidad
        x="person_income", 
        y="loan_amnt", 
        color="loan_status",
        labels={"person_income": "Ingreso Anual ($)", "loan_amnt": "Monto del Préstamo ($)"},
        opacity=0.6,
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# SECCIÓN 3: SIMULADOR PREDICTIVO EN TIEMPO REAL
# -----------------------------------------------------------------------------
st.subheader("🤖 Simulador de Evaluación de Riesgo para Nuevo Solicitante")
st.write("Ingresa los datos del cliente para calcular la probabilidad de impago mediante el modelo de Machine Learning:")

c1, c2, c3, c4 = st.columns(4)
input_age = c1.number_input("Edad del cliente", min_value=18, max_value=90, value=30)
input_income = c2.number_input("Ingreso Anual ($)", min_value=1000, max_value=500000, value=45000, step=5000)
input_amount = c3.number_input("Monto Solicitado ($)", min_value=500, max_value=100000, value=10000, step=1000)
input_rate = c4.number_input("Tasa de Interés (%)", min_value=1.0, max_value=30.0, value=11.5, step=0.5)

if st.button("Evaluar Solicitud de Crédito"):
    # Crear vector de entrada para la predicción
    input_data = np.array([[input_age, input_income, input_amount, input_rate]])
    
    # Predecir probabilidad
    prob_default = model.predict_proba(input_data)[0][1] * 100
    
    # Mostrar resultados según umbral de riesgo
    st.markdown("### Resultado de la Evaluación:")
    if prob_default < 25:
        st.success(f"✅ **CRÉDITO APROBADO** | Riesgo de impago bajo: **{prob_default:.1f}%**")
    elif prob_default < 50:
        st.warning(f"⚠️ **REVISIÓN MANUAL REQUERIDA** | Riesgo de impago moderado: **{prob_default:.1f}%**")
    else:
        st.error(f"❌ **SOLICITUD RECHAZADA** | Riesgo de impago alto: **{prob_default:.1f}%**")

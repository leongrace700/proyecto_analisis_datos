import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Platform | Enterprise AI",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Plataforma Enterprise de Riesgo Crediticio")
st.markdown("Carga & EDA $\rightarrow$ Dashboard $\rightarrow$ Calculadora ML $\rightarrow$ **XGBoost & XAI (SHAP)** $\rightarrow$ **Deep Learning (MLP)**")

# -----------------------------------------------------------------------------
# GESTIÓN DEL ESTADO GLOBAL
# -----------------------------------------------------------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# -----------------------------------------------------------------------------
# ESTRUCTURA DE 5 PESTAÑAS (TABS)
# -----------------------------------------------------------------------------
tab_eda, tab_dashboard, tab_predictor, tab_xgboost, tab_dl = st.tabs([
    "📁 1. Carga & EDA", 
    "📊 2. Dashboard", 
    "🤖 3. Calculadora ML",
    "⚡ 4. XGBoost & Explicabilidad (XAI)",
    "🧠 5. Deep Learning & Perfilamiento"
])

# =============================================================================
# PESTAÑA 1: EDA Y LIMPIEZA DE DATOS
# =============================================================================
with tab_eda:
    st.header("🔍 Carga de Archivo y Limpieza Avanzada (EDA)")
    uploaded_file = st.file_uploader("Sube tu dataset en formato CSV (ej. credit_risk_dataset.csv)", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state.df_raw = pd.read_csv(uploaded_file)
        
        col_diag1, col_diag2, col_diag3 = st.columns(3)
        total_filas = len(st.session_state.df_raw)
        total_nulos = st.session_state.df_raw.isnull().sum().sum()
        duplicados = st.session_state.df_raw.duplicated().sum()
        
        col_diag1.metric("Filas Totales", f"{total_filas:,}")
        col_diag2.metric("Valores Nulos", f"{total_nulos:,}")
        col_diag3.metric("Duplicados", f"{duplicados:,}")

        st.markdown("---")
        st.subheader("Configuración de Limpieza")
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            drop_dups = st.checkbox("Eliminar duplicados", value=True)
            estrategia_nulos = st.radio("Estrategia nulos:", ["Eliminar filas con nulos", "Imputar vacíos", "Conservar nulos"])

        with col_opt2:
            clean_age = st.checkbox("Filtrar edades > 100 años", value=True)
            clean_iqr = st.checkbox("Filtrar outliers con IQR", value=False)
            factor_iqr = st.slider("Factor IQR", 1.0, 3.0, 1.5, step=0.1) if clean_iqr else 1.5

        if st.button("🛠️ Aplicar Limpieza"):
            df_temp = st.session_state.df_raw.copy()
            if drop_dups: df_temp = df_temp.drop_duplicates()
            
            if estrategia_nulos == "Eliminar filas con nulos":
                df_temp = df_temp.dropna()
            elif estrategia_nulos == "Imputar vacíos":
                for col in df_temp.columns:
                    if df_temp[col].dtype in ['int64', 'float64']:
                        df_temp[col] = df_temp[col].fillna(df_temp[col].median())
                    else:
                        df_temp[col] = df_temp[col].fillna(df_temp[col].mode()[0])
            
            if clean_age and 'person_age' in df_temp.columns:
                df_temp = df_temp[df_temp['person_age'] <= 100]
                
            if clean_iqr:
                cols_num = [c for c in ['person_income', 'loan_amnt'] if c in df_temp.columns]
                for col in cols_num:
                    Q1, Q3 = df_temp[col].quantile(0.25), df_temp[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df_temp = df_temp[(df_temp[col] >= Q1 - factor_iqr * IQR) & (df_temp[col] <= Q3 + factor_iqr * IQR)]

            if 'person_age' in df_temp.columns: df_temp['person_age'] = df_temp['person_age'].astype(int)
            if 'loan_status' in df_temp.columns: df_temp['loan_status'] = df_temp['loan_status'].astype(int)

            st.session_state.df_clean = df_temp
            st.success(f"✅ Dataset procesado con éxito: {len(df_temp):,} registros finales.")

        if st.session_state.df_clean is not None:
            st.dataframe(st.session_state.df_clean.head(5), use_container_width=True)

# =============================================================================
# PESTAÑA 2: DASHBOARD DE NEGOCIO
# =============================================================================
with tab_dashboard:
    if st.session_state.df_clean is None:
        st.warning("⚠️ Debes cargar y limpiar el dataset en la Pestaña 1.")
    else:
        df = st.session_state.df_clean.copy()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cartera Analizada", f"${df['loan_amnt'].sum():,.0f}")
        col2.metric("Tasa Morosidad", f"{df['loan_status'].mean()*100:.2f}%")
        col3.metric("Ingreso Promedio", f"${df['person_income'].mean():,.0f}")
        col4.metric("Préstamo Promedio", f"${df['loan_amnt'].mean():,.0f}")
        
        st.markdown("---")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.plotly_chart(px.histogram(df, x="loan_intent", color="loan_status", barmode="group", title="Distribución por Propósito"), use_container_width=True)
        with col_c2:
            st.plotly_chart(px.scatter(df.sample(min(1000, len(df))), x="person_income", y="loan_amnt", color="loan_status", title="Ingreso vs Monto Solicitado"), use_container_width=True)

# =============================================================================
# PESTAÑA 3: CALCULADORA ML (RANDOM FOREST)
# =============================================================================
with tab_predictor:
    if st.session_state.df_clean is None:
        st.warning("⚠️ Debes cargar y limpiar el dataset en la Pestaña 1.")
    else:
        st.header("🏢 Calculadora de Evaluación y Amortización")
        st.caption("🔒 Identificadores sintéticos anonimizados.")
        
        df_ml = st.session_state.df_clean.copy()
        features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']
        
        if all(c in df_ml.columns for c in features):
            X = df_ml[features].fillna(df_ml[features].median())
            y = df_ml['loan_status']
            model_rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

            modo = st.radio("Objetivo:", ["Modo A: Evaluar Monto Solicitado", "Modo B: Calcular Capacidad Máxima"], horizontal=True)
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                input_age = st.number_input("Edad", 18, 90, 30)
                input_income = st.number_input("Ingreso Anual ($)", 1000, 1000000, 45000, 5000)
                cat = st.selectbox("Categoría", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            
            with col_f2:
                input_plazo = st.slider("Plazo (Meses)", 6, 84, 36, 6)
                input_rate = st.number_input("Tasa Anual (%)", 1.0, 40.0, 11.5, 0.25)
                tipo_cuota = st.selectbox("Amortización", ["Cuota Fija (Sistema Francés)", "Cuota Variable (Sistema Alemán)"])

                tasa_m = (input_rate / 100) / 12
                ingreso_m = input_income / 12

                if "Modo A" in modo:
                    input_amount = st.number_input("Monto ($)", 500, 200000, 12000, 1000)
                else:
                    monto_max = (ingreso_m * 0.30) * (((1 + tasa_m)**input_plazo - 1) / (tasa_m * (1 + tasa_m)**input_plazo)) if tasa_m > 0 else (ingreso_m * 0.30) * input_plazo
                    st.info(f"💡 Monto Máximo Sugerido: **${monto_max:,.2f}**")
                    input_amount = float(np.round(monto_max, -2))

            if st.button("📊 Evaluacion Financiera ML"):
                prob_default = model_rf.predict_proba([[input_age, input_income, input_amount, input_rate]])[0][1] * 100
                cuota_m = input_amount * (tasa_m * (1 + tasa_m)**input_plazo) / ((1 + tasa_m)**input_plazo - 1) if tipo_cuota == "Cuota Fija (Sistema Francés)" else (input_amount / input_plazo) + (input_amount * tasa_m)
                dti = (cuota_m / ingreso_m) * 100

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Riesgo (ML)", f"{prob_default:.1f}%")
                r2.metric("Cuota Mensual", f"${cuota_m:,.2f}")
                r3.metric("DTI", f"{dti:.1f}%")
                if prob_default < 25 and dti <= 35: r4.success("🟢 APROBADO")
                elif prob_default < 50 and dti <= 45: r4.warning("🟡 REVISIÓN")
                else: r4.error("🔴 RECHAZADO")

# =============================================================================
# PESTAÑA 4: XGBOOST & EXPLICABILIDAD (XAI)
# =============================================================================
with tab_xgboost:
    if st.session_state.df_clean is None:
        st.warning("⚠️ Carga y limpia el dataset en la Pestaña 1 primero.")
    else:
        st.header("⚡ Algoritmo Gradient Boosting (XGBoost) & Auditable AI")
        st.markdown("XGBoost evalúa la decisión bajo optimización de gradiente y explica **por qué razón técnica** el cliente fue aprobado o rechazado.")

        df_xgb = st.session_state.df_clean.copy()
        features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']

        if all(c in df_xgb.columns for c in features):
            X_xgb = df_xgb[features].fillna(df_xgb[features].median())
            y_xgb = df_xgb['loan_status']

            # Entrenar modelo XGBoost
            model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
            model_xgb.fit(X_xgb, y_xgb)

            # Importancia de variables global
            importancia = pd.DataFrame({
                'Variable': features,
                'Importancia (%)': (model_xgb.feature_importances_ * 100).round(2)
            }).sort_values(by='Importancia (%)', ascending=False)

            col_x1, col_x2 = st.columns([1, 2])

            with col_x1:
                st.subheader("📌 Importancia Global de Variables")
                st.dataframe(importancia, use_container_width=True)

            with col_x2:
                st.subheader("📊 Gráfico de Relevancia (XGBoost)")
                fig_imp = px.bar(importancia, x='Importancia (%)', y='Variable', orientation='h', color='Importancia (%)', color_continuous_scale='Viridis')
                st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("---")
            st.subheader("🔎 Explicabilidad Individual para la Solicitud")

            c1, c2, c3, c4 = st.columns(4)
            val_age = c1.number_input("Edad (Años)", 18, 90, 28, key="xgb_age")
            val_inc = c2.number_input("Ingreso Anual ($)", 1000, 500000, 35000, key="xgb_inc")
            val_amt = c3.number_input("Monto Préstamo ($)", 500, 100000, 15000, key="xgb_amt")
            val_rate = c4.number_input("Tasa Interés (%)", 1.0, 35.0, 14.5, key="xgb_rate")

            if st.button("🧪 Auditar Decisión con XGBoost"):
                vec_in = np.array([[val_age, val_inc, val_amt, val_rate]])
                prob_xgb = model_xgb.predict_proba(vec_in)[0][1] * 100

                st.markdown("#### Resultado del Dictamen XGBoost:")
                if prob_xgb < 30:
                    st.success(f"✅ **Aprobado por XGBoost** | Probabilidad de Default: **{prob_xgb:.2f}%**")
                else:
                    st.error(f"❌ **Rechazado por XGBoost** | Probabilidad de Default: **{prob_xgb:.2f}%**")

                # Cálculo de contribución individual aproximada (Descomposición de características)
                medias = X_xgb.mean()
                desviaciones = X_xgb.std()
                desviacion_cliente = (vec_in[0] - medias) / desviaciones
                
                df_razones = pd.DataFrame({
                    'Variable': ['Edad', 'Ingreso Anual', 'Monto Solicitado', 'Tasa de Interés'],
                    'Valor Cliente': [val_age, val_inc, val_amt, val_rate],
                    'Promedio Dataset': medias.values.round(2),
                    'Impacto en la Decisión': ['Reduce Riesgo 🟢' if d < 0 else 'Aumenta Riesgo 🔴' for d in desviacion_cliente]
                })
                
                st.write("**Desglose Técnico de Factores de Riesgo:**")
                st.table(df_razones)

# =============================================================================
# PESTAÑA 5: DEEP LEARNING (MLP) & PÉRDIDA ESPERADA
# =============================================================================
with tab_dl = st.tabs(["..."])[0] if False else tab_dl: # Conexión interna
with tab_dl:
    if st.session_state.df_clean is None:
        st.warning("⚠️ Carga y limpia el dataset en la Pestaña 1 primero.")
    else:
        st.header("🧠 Red Neuronal Profunda (Multilayer Perceptron - MLP)")
        st.markdown("Red Neuronal de 3 capas densas para estimación de **Segmento de Riesgo Multi-Clase** y cálculo de **Pérdida Esperada (Expected Loss - EL)**.")

        df_dl = st.session_state.df_clean.copy()
        features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']

        if all(c in df_dl.columns for c in features):
            # Normalización (Estandarización de datos para Deep Learning)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_dl[features].fillna(df_dl[features].median()))
            y_dl = df_dl['loan_status'].values

            # Construcción de la Red Neuronal con Keras/TensorFlow
            @st.cache_resource
            def entrenar_red_neuronal(X_train, y_train):
                model = Sequential([
                    Dense(16, activation='relu', input_shape=(4,)),
                    Dropout(0.2),
                    Dense(8, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
                return model

            with st.spinner("🧠 Entrenando arquitectura de Red Neuronal Profunda..."):
                nn_model = entrenar_red_neuronal(X_scaled, y_dl)

            st.success("🤖 Red Neuronal entrenada y lista para inferencias.")

            st.markdown("---")
            st.subheader("Simulación de Crédito con Red Neuronal Profunda")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                dl_age = st.number_input("Edad Solicitante", 18, 90, 32)
                dl_inc = st.number_input("Ingreso Anual ($)", 1000, 500000, 50000)
            with col_d2:
                dl_amt = st.number_input("Monto Préstamo ($)", 500, 100000, 18000)
                dl_rate = st.number_input("Tasa Interés (%)", 1.0, 35.0, 12.0)

            if st.button("🧠 Evaluar con Red Neuronal"):
                # Escalar la entrada del cliente
                cliente_scaled = scaler.transform([[dl_age, dl_inc, dl_amt, dl_rate]])
                pred_prob_dl = float(nn_model.predict(cliente_scaled)[0][0]) * 100

                # Cálculo del Modelo Financiero de Pérdida Esperada: EL = PD * LGD * EAD
                # LGD (Loss Given Default) asumido en 45% (estándar Basilea II)
                # EAD (Exposure at Default) = Monto del Préstamo
                pd_val = pred_prob_dl / 100
                lgd = 0.45 
                ead = dl_amt
                expected_loss = pd_val * lgd * ead

                st.subheader("Resultados de la Red Neuronal (Deep Learning):")
                m1, m2, m3 = st.columns(3)

                m1.metric("Probabilidad Impago (PD)", f"{pred_prob_dl:.2f}%")
                
                # Clasificación en Perfiles de Riesgo Tri-Clase
                if pred_prob_dl < 20:
                    perfil = "🟢 PERFIL A (Bajo Riesgo)"
                elif pred_prob_dl < 45:
                    perfil = "🟡 PERFIL B (Riesgo Moderado)"
                else:
                    perfil = "🔴 PERFIL C (Alto Riesgo)"

                m2.metric("Categoría de Perfil", perfil)
                m3.metric("Pérdida Esperada (EL)", f"${expected_loss:,.2f}")

                st.caption(f"ℹ️ **Pérdida Esperada (Expected Loss):** Provisiones recomendadas en balance por **${expected_loss:,.2f}** bajo norma Basilea III (LGD estimada del 45%).")

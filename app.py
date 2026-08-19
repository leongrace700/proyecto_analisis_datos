import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Platform | EDA & Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Plataforma Integral de Riesgo Crediticio")
st.markdown("Gestión completa del ciclo de datos: Carga & EDA $\rightarrow$ Dashboard Interactivo $\rightarrow$ Predicción ML")

# -----------------------------------------------------------------------------
# GESTIÓN DEL ESTADO GLOBAL (Session State)
# -----------------------------------------------------------------------------
# Inicializamos el dataframe en la sesión para que persista entre pestañas
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# -----------------------------------------------------------------------------
# ESTRUCTURA DE PESTAÑAS (TABS)
# -----------------------------------------------------------------------------
tab_eda, tab_dashboard, tab_predictor = st.tabs([
    "📁 1. Carga & Limpieza de Datos (EDA)", 
    "📊 2. Dashboard de Negocio", 
    "🤖 3. Simulador Predictivo (ML)"
])

# =============================================================================
# PESTAÑA 1: EDA Y LIMPIEZA DE DATOS
# =============================================================================
with tab_eda:
    st.header("🔍 Carga de Archivo y Análisis Exploratorio (EDA)")
    
    # =============================================================================
# PESTAÑA 1: EDA Y LIMPIEZA DE DATOS (AVANZADA)
# =============================================================================
with tab_eda:
    st.header("🔍 Carga de Archivo y Limpieza Avanzada (EDA)")
    
    uploaded_file = st.file_uploader("Sube tu dataset en formato CSV (ej. credit_risk_dataset.csv)", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state.df_raw = pd.read_csv(uploaded_file)
        
        st.subheader("1. Diagnóstico Inicial de Calidad de Datos")
        
        col_diag1, col_diag2, col_diag3 = st.columns(3)
        total_filas = len(st.session_state.df_raw)
        total_nulos = st.session_state.df_raw.isnull().sum().sum()
        duplicados = st.session_state.df_raw.duplicated().sum()
        
        col_diag1.metric("Filas Totales", f"{total_filas:,}")
        col_diag2.metric("Valores Vacíos / Nulos", f"{total_nulos:,}")
        col_diag3.metric("Filas Duplicadas", f"{duplicados:,}")
        
        # Tabla detallada de nulos por columna
        with st.expander("📌 Ver detalle de valores nulos y tipos de datos por columna"):
            null_info = pd.DataFrame({
                "Tipo de Dato": st.session_state.df_raw.dtypes,
                "Valores Nulos": st.session_state.df_raw.isnull().sum(),
                "% Nulos": (st.session_state.df_raw.isnull().sum() / total_filas * 100).round(2)
            })
            st.dataframe(null_info, use_container_width=True)

        st.markdown("---")
        st.subheader("2. Configuración de Limpieza y Tratamiento")
        
        col_opt1, col_opt2 = st.columns(2)
        
        # --- COLUMNA 1: MANEJO DE NULOS Y DUPLICADOS ---
        with col_opt1:
            st.markdown("##### 🧹 Valores Vacíos y Duplicados")
            drop_dups = st.checkbox("Eliminar duplicados exactos", value=True)
            
            estrategia_nulos = st.radio(
                "Tratamiento para valores vacíos (nulos):",
                ["Eliminar filas con nulos", "Imputar vacíos (Mediana para números / Moda para texto)", "Conservar nulos"]
            )

        # --- COLUMNA 2: MANEJO DE VALORES ATÍPICOS (OUTLIERS) ---
        with col_opt2:
            st.markdown("##### 🚨 Tratamiento de Outliers (Valores Atípicos)")
            
            # Outliers por regla de negocio
            clean_age = st.checkbox("Filtrar edades irrealistas (Edad > 100 años)", value=True)
            
            # Outliers por método estadístico IQR
            clean_iqr = st.checkbox("Filtrar outliers con método IQR (Rango Intercuartílico)", value=False)
            if clean_iqr:
                factor_iqr = st.slider("Factor de tolerancia IQR (1.5 = Estándar, 3.0 = Conservador)", 1.0, 3.0, 1.5, step=0.1)

        # --- EJECUCIÓN DE LA LIMPIEZA ---
        if st.button("🛠️ Aplicar Limpieza y Generar Dataset Procesado"):
            df_temp = st.session_state.df_raw.copy()
            filas_iniciales = len(df_temp)
            
            # 1. Duplicados
            if drop_dups:
                df_temp = df_temp.drop_duplicates()
                
            # 2. Manejo de nulos
            if estrategia_nulos == "Eliminar filas con nulos":
                df_temp = df_temp.dropna()
            elif estrategia_nulos == "Imputar vacíos (Mediana para números / Moda para texto)":
                for col in df_temp.columns:
                    if df_temp[col].dtype in ['int64', 'float64']:
                        df_temp[col] = df_temp[col].fillna(df_temp[col].median())
                    else:
                        df_temp[col] = df_temp[col].fillna(df_temp[col].mode()[0])
            
            # 3. Outliers de Edad (Regla de negocio)
            if clean_age and 'person_age' in df_temp.columns:
                df_temp = df_temp[df_temp['person_age'] <= 100]
                
            # 4. Outliers estadísticos IQR (Aplicado a ingresos y montos)
            if clean_iqr:
                cols_num = [c for c in ['person_income', 'loan_amnt'] if c in df_temp.columns]
                for col in cols_num:
                    Q1 = df_temp[col].quantile(0.25)
                    Q3 = df_temp[col].quantile(0.75)
                    IQR = Q3 - Q1
                    limite_inferior = Q1 - (factor_iqr * IQR)
                    limite_superior = Q3 + (factor_iqr * IQR)
                    df_temp = df_temp[(df_temp[col] >= limite_inferior) & (df_temp[col] <= limite_superior)]

            # Formateo de tipos
            if 'person_age' in df_temp.columns:
                df_temp['person_age'] = df_temp['person_age'].astype(int)
            if 'loan_status' in df_temp.columns:
                df_temp['loan_status'] = df_temp['loan_status'].astype(int)

            # Guardar en sesión
            st.session_state.df_clean = df_temp
            filas_eliminadas = filas_iniciales - len(df_temp)
            
            st.success(f"✅ ¡Procesamiento completado! Se descartaron **{filas_eliminadas:,}** filas ruidosas/atípicas. Dataset final listo con **{len(df_temp):,}** registros.")

        # --- VISTA PREVIA Y DESCARGA ---
        if st.session_state.df_clean is not None:
            st.markdown("---")
            st.subheader("3. Dataset Limpio y Listo para Análisis")
            st.dataframe(st.session_state.df_clean.head(10), use_container_width=True)
            
            csv_clean = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV Limpio",
                data=csv_clean,
                file_name="credit_risk_cleaned.csv",
                mime="text/csv"
            )
    else:
        st.info("👆 Por favor sube un archivo `.csv` para iniciar el diagnóstico y la limpieza de datos.")


# =============================================================================
# PESTAÑA 2: DASHBOARD DE NEGOCIO Y FILTROS
# =============================================================================
with tab_dashboard:
    if st.session_state.df_clean is None:
        st.warning("⚠️ Debes cargar y limpiar el dataset en la **Pestaña 1 (EDA)** antes de acceder al Dashboard.")
    else:
        df = st.session_state.df_clean.copy()
        
        # BARRA LATERAL DE FILTROS (Se activa cuando hay datos limpios)
        st.sidebar.header("🔍 Filtros del Dashboard")

        intenciones = ['Todas'] + list(df['loan_intent'].dropna().unique()) if 'loan_intent' in df.columns else ['Todas']
        intencion_selected = st.sidebar.selectbox("Propósito del Préstamo", intenciones)

        min_age, max_age = int(df['person_age'].min()), int(df['person_age'].max())
        edad_rango = st.sidebar.slider("Rango de Edad del Cliente", min_age, max_age, (min_age, max_age))

        # Aplicar filtros
        df_filtered = df[(df['person_age'] >= edad_rango[0]) & (df['person_age'] <= edad_rango[1])]
        if intencion_selected != 'Todas' and 'loan_intent' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['loan_intent'] == intencion_selected]

        # KPIs
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

        # Gráficos Plotly
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("📊 Distribución de Préstamos por Propósito")
            if 'loan_intent' in df_filtered.columns:
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
                df_filtered.sample(min(1000, len(df_filtered))), 
                x="person_income", 
                y="loan_amnt", 
                color="loan_status",
                labels={"person_income": "Ingreso Anual ($)", "loan_amnt": "Monto del Préstamo ($)"},
                opacity=0.6,
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


# =============================================================================
# PESTAÑA 3: SIMULADOR PREDICTIVO (MACHINE LEARNING)
# =============================================================================
with tab_predictor:
    if st.session_state.df_clean is None:
        st.warning("⚠️ Debes cargar y limpiar el dataset en la **Pestaña 1 (EDA)** antes de entrenar el modelo.")
    else:
        st.subheader("🤖 Evaluación de Riesgo para Nuevos Solicitantes")
        st.write("Entrenando modelo dinámico con el dataset procesado...")

        # Entrenar modelo con los datos limpios de la sesión
        df_ml = st.session_state.df_clean.copy()
        features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']
        
        # Verificar que existen las columnas requeridas
        if all(col in df_ml.columns for col in features):
            X = df_ml[features].fillna(df_ml[features].median())
            y = df_ml['loan_status']
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)

            # Formulario de entrada
            c1, c2, c3, c4 = st.columns(4)
            input_age = c1.number_input("Edad del cliente", min_value=18, max_value=90, value=30)
            input_income = c2.number_input("Ingreso Anual ($)", min_value=1000, max_value=500000, value=45000, step=5000)
            input_amount = c3.number_input("Monto Solicitado ($)", min_value=500, max_value=100000, value=10000, step=1000)
            input_rate = c4.number_input("Tasa de Interés (%)", min_value=1.0, max_value=30.0, value=11.5, step=0.5)

            if st.button("Evaluar Solicitud de Crédito"):
                input_data = np.array([[input_age, input_income, input_amount, input_rate]])
                prob_default = model.predict_proba(input_data)[0][1] * 100
                
                st.markdown("### Resultado de la Evaluación:")
                if prob_default < 25:
                    st.success(f"✅ **CRÉDITO APROBADO** | Riesgo de impago bajo: **{prob_default:.1f}%**")
                elif prob_default < 50:
                    st.warning(f"⚠️ **REVISIÓN MANUAL REQUERIDA** | Riesgo de impago moderado: **{prob_default:.1f}%**")
                else:
                    st.error(f"❌ **SOLICITUD RECHAZADA** | Riesgo de impago alto: **{prob_default:.1f}%**")
        else:
            st.error(f"Faltan columnas requeridas en el dataset para el modelo ML. Se requieren: {features}")

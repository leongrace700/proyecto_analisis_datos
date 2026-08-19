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
# PESTAÑA 3: SIMULADOR Y CALCULADORA CREDITICIA DE RIESGO (ENFOQUE CORPORATIVO)
# =============================================================================
with tab_predictor:
    if st.session_state.df_clean is None:
        st.warning("⚠️ Debes cargar y limpiar el dataset en la **Pestaña 1 (EDA)** antes de acceder a la Calculadora de Riesgo.")
    else:
        st.header("🏢 Calculadora de Evaluación y Amortización para Analistas de Riesgo")
        st.markdown("Herramienta de decisión interna para estructuración de créditos y estimación de riesgo de incumplimiento.")

        # Entrenar modelo con los datos limpios de la sesión
        df_ml = st.session_state.df_clean.copy()
        features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']
        
        if all(col in df_ml.columns for col in features):
            X = df_ml[features].fillna(df_ml[features].median())
            y = df_ml['loan_status']
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)

            st.markdown("---")
            st.subheader("1. Parámetros del Solicitante y Estructuración del Crédito")

            # --- FORMULARIO EN COLUMNAS ---
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                st.markdown("##### 👤 Perfil del Solicitante")
                input_age = st.number_input("Edad del Solicitante (Años)", min_value=18, max_value=90, value=30)
                input_income = st.number_input("Ingreso Anual Verificado ($)", min_value=1000, max_value=1000000, value=45000, step=5000)
                
                # Categoría / Propósito del Préstamo (Loan Intent)
                categorias_credito = {
                    "PERSONAL": {"plazo_std": 36, "tasa_promedio": 11.5},
                    "EDUCATION": {"plazo_std": 48, "tasa_promedio": 9.8},
                    "MEDICAL": {"plazo_std": 24, "tasa_promedio": 12.0},
                    "VENTURE": {"plazo_std": 60, "tasa_promedio": 14.5},
                    "HOMEIMPROVEMENT": {"plazo_std": 60, "tasa_promedio": 10.5},
                    "DEBTCONSOLIDATION": {"plazo_std": 48, "tasa_promedio": 13.2}
                }
                
                categoria_sel = st.selectbox("Categoría / Destino del Crédito", list(categorias_credito.keys()))

            with col_f2:
                st.markdown("##### ⚙️ Condiciones Financieras del Préstamo")
                input_amount = st.number_input("Monto Solicitado ($)", min_value=500, max_value=200000, value=12000, step=1000)
                
                # Plazo estimado / sugerido según categoría de crédito
                plazo_sugerido = categorias_credito[categoria_sel]["plazo_std"]
                input_plazo_meses = st.slider("Plazo del Crédito (Meses)", 6, 84, plazo_sugerido, step=6)
                
                # Tasa de interés anual
                tasa_sugerida = categorias_credito[categoria_sel]["tasa_promedio"]
                input_rate = st.number_input("Tasa de Interés Anual Efec. (%)", min_value=1.0, max_value=40.0, value=tasa_sugerida, step=0.25)
                
                # Tipo de Sistema de Amortización (Tipo de Cuota)
                tipo_cuota = st.selectbox("Sistema de Amortización (Tipo de Cuota)", ["Cuota Fija (Sistema Francés)", "Cuota Variable (Sistema Alemán - Abono Fijo a Capital)"])

            st.markdown("---")
            
            # --- BOTÓN DE EVALUACIÓN Y CÁLCULO ---
            if st.button("📊 Generar Dictamen Financiero y Corrida de Crédito"):
                
                # 1. EVALUACIÓN DE MACHINE LEARNING (PREDICCIÓN DE RIESGO)
                input_data = np.array([[input_age, input_income, input_amount, input_rate]])
                prob_default = model.predict_proba(input_data)[0][1] * 100
                
                # 2. CÁLCULOS FINANCIEROS INTERNOS
                tasa_mensual = (input_rate / 100) / 12
                ingreso_mensual = input_income / 12
                
                # Cálculo según tipo de cuota
                if tipo_cuota == "Cuota Fija (Sistema Francés)":
                    if tasa_mensual > 0:
                        cuota_mensual_inicial = input_amount * (tasa_mensual * (1 + tasa_mensual)**input_plazo_meses) / ((1 + tasa_mensual)**input_plazo_meses - 1)
                    else:
                        cuota_mensual_inicial = input_amount / input_plazo_meses
                    cuota_texto = f"${cuota_mensual_inicial:,.2f} (Fija)"
                else:
                    # Sistema Alemán: Abono constante a capital + Intereses sobre saldo
                    abono_capital = input_amount / input_plazo_meses
                    interes_mes_1 = input_amount * tasa_mensual
                    cuota_mensual_inicial = abono_capital + interes_mes_1
                    cuota_texto = f"${cuota_mensual_inicial:,.2f} (Primera Cuota - Decreciente)"

                # Ratio de Capacidad de Pago (DTI - Debt to Income Ratio Mensual)
                dti_ratio = (cuota_mensual_inicial / ingreso_mensual) * 100

                # --- MUESTRA DE RESULTADOS GERENCIALES ---
                st.subheader("2. Dictamen Técnico y Viabilidad del Crédito")
                
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                
                res_col1.metric("Probabilidad de Impago (ML)", f"{prob_default:.1f}%")
                res_col2.metric("Cuota Mensual Estimada", cuota_texto)
                res_col3.metric("Relación Deuda/Ingreso (DTI)", f"{dti_ratio:.1f}%")
                
                # Estado del dictamen según reglas combinadas de riesgo y capacidad de pago
                if prob_default < 25 and dti_ratio <= 35:
                    res_col4.success("🟢 **RECOMENDACIÓN: APROBADO**")
                elif prob_default < 50 and dti_ratio <= 45:
                    res_col4.warning("🟡 **RECOMENDACIÓN: REVISIÓN DE COMITÉ**")
                else:
                    res_col4.error("🔴 **RECOMENDACIÓN: RECHAZADO**")

                # Observaciones técnicas para la mesa de control
                with st.expander("📋 Ver Observaciones del Comité de Riesgos"):
                    st.write(f"• **Relación Deuda/Ingreso (DTI):** Compromete el **{dti_ratio:.1f}%** del ingreso mensual del solicitante. *(Límite aconsejable: 35%)*.")
                    st.write(f"• **Score de Riesgo Predictivo:** El algoritmo ubica la probabilidad de mora en **{prob_default:.1f}%**.")
                    st.write(f"• **Plazo Asignado:** **{input_plazo_meses} meses** adaptado para la categoría **{categoria_sel}**.")

                # --- TABLA DE AMORTIZACIÓN (CORRIDA FINANCIERA) ---
                st.markdown("---")
                st.subheader("3. Tabla de Amortización Proyectada")
                
                # Generación de la tabla mes a mes
                cronograma = []
                saldo_pendiente = input_amount
                
                for mes in range(1, input_plazo_meses + 1):
                    interes_periodo = saldo_pendiente * tasa_mensual
                    
                    if tipo_cuota == "Cuota Fija (Sistema Francés)":
                        cuota_periodo = cuota_mensual_inicial
                        capital_periodo = cuota_periodo - interes_periodo
                    else:
                        capital_periodo = input_amount / input_plazo_meses
                        cuota_periodo = capital_periodo + interes_periodo
                        
                    saldo_pendiente -= capital_periodo
                    if saldo_pendiente < 0: 
                        saldo_pendiente = 0
                        
                    cronograma.append({
                        "Mes": mes,
                        "Cuota Total ($)": round(cuota_periodo, 2),
                        "Abono a Capital ($)": round(capital_periodo, 2),
                        "Intereses ($)": round(interes_periodo, 2),
                        "Saldo Pendiente ($)": round(saldo_pendiente, 2)
                    })
                
                df_amortizacion = pd.DataFrame(cronograma)
                
                # Graficar la composición de pagos
                fig_amort = px.bar(
                    df_amortizacion, 
                    x="Mes", 
                    y=["Abono a Capital ($)", "Intereses ($)"],
                    title="Evolución de Capital vs. Intereses a lo Largo del Plazo",
                    barmode="stack",
                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                )
                st.plotly_chart(fig_amort, use_container_width=True)
                
                # Mostrar tabla completa con opción a descarga
                st.dataframe(df_amortizacion, use_container_width=True)
                
                csv_amort = df_amortizacion.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Exportar Tabla de Amortización (CSV)",
                    data=csv_amort,
                    file_name=f"amortizacion_{categoria_sel}_{input_amount}.csv",
                    mime="text/csv"
                )

        else:
            st.error(f"Faltan columnas requeridas en el dataset para el modelo ML. Se requieren: {features}")

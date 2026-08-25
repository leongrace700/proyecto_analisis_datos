import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y TEMA CLARO
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Monitor Financiero & Cotizador de Créditos Colombia",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #F8F9FA;
        color: #212529;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #0F52BA;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #E9ECEF;
        padding: 8px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #495057;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F52BA !important;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.08);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE CÁLCULO FINANCIERO
# -----------------------------------------------------------------------------
def ea_to_em(tasa_ea):
    if pd.isna(tasa_ea) or tasa_ea <= 0:
        return 0.0
    return ((1 + tasa_ea / 100.0) ** (1.0 / 12.0) - 1.0)

def calcular_cuota_fija(monto, tasa_ea, plazo_meses):
    if plazo_meses <= 0 or monto <= 0:
        return 0.0, 0.0, 0.0
    
    i_m = ea_to_em(tasa_ea)
    if i_m == 0:
        cuota = monto / plazo_meses
    else:
        cuota = monto * (i_m * ((1 + i_m) ** plazo_meses)) / (((1 + i_m) ** plazo_meses) - 1)
    
    total_pagar = cuota * plazo_meses
    total_intereses = total_pagar - monto
    return cuota, total_intereses, total_pagar

# -----------------------------------------------------------------------------
# LIMPIEZA Y TRANSFORMACIÓN DE DATOS DINÁMICA
# -----------------------------------------------------------------------------
def clean_data(df):
    df = df.copy()

    # 1. Normalización inicial de nombres de columnas
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("á", "a")
        .str.replace("é", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
        .str.replace("ñ", "n")
    )

    # 2. Mapeo específico ajustado a los encabezados de Datos Abiertos Colombia / Socrata
    col_map = {
        # Entidad
        'nombre_de_la_entidad': 'nombre_entidad',
        'entidad': 'nombre_entidad',
        'banco': 'nombre_entidad',
        
        # Tipo / Modalidad de Crédito
        'tipo_de_credito': 'tipo_credito',
        'tipo_de_cr_dito': 'tipo_credito',
        'modalidad': 'tipo_credito',
        'linea_de_credito': 'tipo_credito',
        'producto': 'tipo_credito',
        
        # Tasa de Interés
        'tasa_efectiva_promedio_ponderada': 'tasa_efectiva_promedio',
        'tasa_efectiva_promedio': 'tasa_efectiva_promedio',
        'tasa_ea': 'tasa_efectiva_promedio',
        'tasa': 'tasa_efectiva_promedio',
        
        # Montos
        'montos_desembolsados': 'monto_desembolsado',
        'monto_desembolsado': 'monto_desembolsado',
        'monto': 'monto_desembolsado',
        
        # Créditos
        'numero_de_creditos': 'numero_creditos',
        'creditos': 'numero_creditos',

        # Encabezados de Datos Abiertos
        'tipo_de_garantia': 'tipo_garantia',
        'tipo_de_garant_a': 'tipo_garantia',
        'garantia': 'tipo_garantia',

        'producto_de_credito': 'producto_credito',
        'producto_de_cr_dito': 'producto_credito',

        'plazo_de_credito': 'plazo_credito',
        'plazo_de_cr_dito': 'plazo_credito',

        'tamano_de_empresa': 'tamano_empresa',
        'tama_o_de_empresa': 'tamano_empresa'
    }
    
    df = df.rename(columns=col_map)

    # 3. Asegurar columnas esenciales si no venían en el CSV
    if 'tipo_credito' not in df.columns:
        df['tipo_credito'] = 'General'
    else:
        df['tipo_credito'] = df['tipo_credito'].fillna('General')

    if 'nombre_entidad' not in df.columns:
        df['nombre_entidad'] = 'Desconocido'
    else:
        df['nombre_entidad'] = df['nombre_entidad'].fillna('Desconocido')

    # 4. Conversión Numérica
    num_cols = ['tasa_efectiva_promedio', 'monto_desembolsado', 'numero_creditos']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. Filtros de calidad
    if 'tasa_efectiva_promedio' in df.columns:
        df = df[df['tasa_efectiva_promedio'] > 0]
        df = df[(df['tasa_efectiva_promedio'] >= 1.0) & (df['tasa_efectiva_promedio'] <= 100.0)]

    if 'monto_desembolsado' in df.columns:
        df['monto_desembolsado'] = df['monto_desembolsado'].fillna(0)

    return df.drop_duplicates()

# Inicialización del Session State
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None

# -----------------------------------------------------------------------------
# INTERFAZ Y PESTAÑAS
# -----------------------------------------------------------------------------
st.title("🏦 Monitor Financiero & Cotizador de Créditos")

tab_inicio, tab_eda, tab_dashboard, tab_simulador, tab_analysis, tab_ml = st.tabs([
    "🏠 0. Proyecto",
    "🔍 1. Cargar & Explorar Datos",
    "📊 2. Dashboard de Tasas",
    "🧮 3. Calculadora & Comparador",
    "📈 4. Análisis e Interpretación",
    "🤖 5. Predicción con ML"
])

# =============================================================================
# PESTAÑA 0: PRESENTACIÓN DEL PROYECTO / PITCH
# =============================================================================
with tab_inicio:

    # Portada
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0F52BA 0%, #1976D2 100%);
        padding: 35px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        text-align: center;
    ">
        <h1 style="color: white; margin-bottom: 10px;">
            🏦 Monitor Financiero & Cotizador de Créditos
        </h1>
        <p style="font-size: 1.2rem; margin-bottom: 5px;">
            Plataforma inteligente para comparar, analizar y simular créditos
        </p>
        <p style="font-size: 0.95rem;">
            Análisis de datos aplicado a la toma de decisiones financieras
        </p>
    </div>
    """, unsafe_allow_html=True)

   # Integrantes
st.markdown("## 👥 Integrantes del equipo")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 👤 Grace Leon")

with col2:
    st.info("### 👤 Mayerly Roman")

with col3:
    st.info("### 👤 Marco Jimenez")

    st.markdown("---")

    # Objetivo
    st.markdown("## 🎯 Objetivo de la solución")

    st.markdown("""
    El proyecto tiene como objetivo desarrollar una herramienta interactiva
    que permita **analizar y comparar las tasas de interés de diferentes
    entidades financieras**, facilitando la simulación de créditos y
    ayudando al usuario a identificar alternativas potencialmente más
    convenientes según el monto y plazo de financiación.

    La solución transforma datos financieros en información clara y útil
    para apoyar la **toma de decisiones financieras**.
    """)

    # Problema y usuarios
    st.markdown("## 💡 Problema que resolvemos")

    problema_col, usuario_col = st.columns(2)

    with problema_col:
        st.markdown("""
        ### ❓ ¿Qué problema resuelve?

        Comparar créditos entre diferentes entidades puede resultar
        complicado debido a:

        - Diferentes tasas de interés.
        - Diferentes modalidades de crédito.
        - Diferentes plazos y condiciones.
        - Dificultad para calcular el costo real del crédito.
        - Información financiera dispersa.

        Nuestra solución centraliza estos datos y los convierte en
        **comparaciones, cálculos y recomendaciones fáciles de interpretar**.
        """)

    with usuario_col:
        st.markdown("""
        ### 👥 ¿Quiénes serían los usuarios?

        La solución puede estar dirigida a:

        - Personas que desean solicitar un crédito.
        - Clientes que quieren comparar diferentes entidades.
        - Asesores financieros.
        - Pequeñas y medianas empresas.
        - Analistas del sector financiero.
        - Organizaciones que necesitan analizar el comportamiento del mercado
          crediticio.
        """)

    st.markdown("---")

    # Diferenciador
    st.markdown("## 🚀 ¿Qué hace diferente nuestra solución?")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("""
        ### 📊 Datos
        Utiliza datos financieros estructurados para analizar el
        comportamiento de las tasas y las entidades.
        """)

    with d2:
        st.markdown("""
        ### 🧮 Simulación
        Permite estimar cuota mensual, intereses y total a pagar
        según el monto y plazo seleccionado.
        """)

    with d3:
        st.markdown("""
        ### 🤖 Inteligencia
        Incorpora Machine Learning para generar estimaciones y
        apoyar la identificación de alternativas financieras.
        """)

    # Tecnologías
    st.markdown("---")
    st.markdown("## 🛠️ Tecnologías utilizadas")

    tecnologias = [
        ("🐍", "Python", "Lenguaje principal"),
        ("🎨", "Streamlit", "Interfaz web interactiva"),
        ("📊", "Pandas", "Manipulación y análisis de datos"),
        ("📈", "Plotly", "Visualización de datos"),
        ("🤖", "Scikit-learn", "Machine Learning"),
        ("💾", "CSV / Excel", "Fuentes de datos")
    ]

    tech_cols = st.columns(3)

    for i, (icono, nombre, descripcion) in enumerate(tecnologias):
        with tech_cols[i % 3]:
            st.markdown(f"""
            <div style="
                background-color: white;
                padding: 18px;
                border-radius: 10px;
                margin-bottom: 15px;
                border: 1px solid #DEE2E6;
                min-height: 100px;
            ">
                <h3>{icono} {nombre}</h3>
                <p>{descripcion}</p>
            </div>
            """, unsafe_allow_html=True)

    # Funcionalidades
    st.markdown("---")
    st.markdown("## ⚙️ Funcionalidades principales")

    funcionalidades = [
        "📂 Carga y transformación automática de archivos CSV y Excel.",
        "📊 Dashboard interactivo para analizar tasas y comportamiento del mercado.",
        "🧮 Calculadora de cuotas para diferentes montos y plazos.",
        "🏦 Comparación de entidades financieras.",
        "📉 Identificación de tasas más bajas y alternativas potencialmente más económicas.",
        "📈 Análisis e interpretación de los principales indicadores.",
        "🤖 Modelo de Machine Learning basado en Random Forest.",
        "🔮 Predicción y recomendación automática de entidades."
    ]

    col_func1, col_func2 = st.columns(2)

    mitad = len(funcionalidades) // 2

    with col_func1:
        for item in funcionalidades[:mitad]:
            st.markdown(f"- {item}")

    with col_func2:
        for item in funcionalidades[mitad:]:
            st.markdown(f"- {item}")

    # Valor mediante datos e IA
    st.markdown("---")
    st.markdown("## 🧠 ¿Cómo aporta valor mediante análisis de datos e IA?")

    st.markdown("""
    La plataforma convierte información financiera en indicadores y
    recomendaciones que facilitan la toma de decisiones.

    **El análisis de datos permite:**

    - Identificar diferencias entre las tasas de las entidades.
    - Encontrar tendencias y patrones en los datos.
    - Comparar el costo potencial de diferentes alternativas.
    - Visualizar la información de forma sencilla.

    **El componente de Machine Learning permite:**

    - Analizar la relación entre monto, tipo de crédito y entidad.
    - Estimar posibles tasas de interés.
    - Generar una recomendación automática basada en los datos disponibles.

    De esta manera, la solución pasa de ser solamente un dashboard a
    convertirse en una herramienta de **apoyo para la toma de decisiones**.
    """)

    # Modelo de negocio
    st.markdown("---")
    st.markdown("## 💼 ¿Cómo podría convertirse en un producto real?")

    st.markdown("""
    La solución podría evolucionar hacia una plataforma web o aplicación
    financiera donde los usuarios ingresen sus necesidades de financiación
    y reciban una comparación personalizada.

    ### Modelo de producto

    **1. Usuario**
    Ingresa monto, plazo y tipo de crédito.

    **2. Plataforma**
    Analiza las tasas y condiciones disponibles.

    **3. Comparador**
    Calcula cuotas, intereses y costo total estimado.

    **4. Recomendador**
    Identifica las alternativas más convenientes.

    **5. Valor comercial**
    La plataforma podría ofrecerse como servicio para consumidores,
    asesores financieros, empresas o entidades interesadas en análisis
    del mercado crediticio.
    """)

    # Flujo de solución
    st.markdown("---")
    st.markdown("## 🔄 ¿Cómo funciona nuestra solución?")

    flujo1, flujo2, flujo3, flujo4 = st.columns(4)

    with flujo1:
        st.markdown("""
        ### 1️⃣ Datos
        Carga de información financiera.
        """)

    with flujo2:
        st.markdown("""
        ### 2️⃣ Análisis
        Limpieza, transformación y análisis.
        """)

    with flujo3:
        st.markdown("""
        ### 3️⃣ Simulación
        Cálculo y comparación de créditos.
        """)

    with flujo4:
        st.markdown("""
        ### 4️⃣ Decisión
        Recomendaciones basadas en datos.
        """)

    # Evidencia visual
    st.markdown("---")
    st.markdown("## 🖥️ Evidencia visual de la solución")

    st.info("""
    📌 **Esta sección puede utilizarse para presentar capturas de pantalla
    de las diferentes funcionalidades de la aplicación.**

    Se recomienda incluir posteriormente:

    - Captura del Dashboard.
    - Captura de la Calculadora & Comparador.
    - Captura del análisis financiero.
    - Captura del modelo predictivo.
    """)

    # Cierre del pitch
    st.markdown("---")

    st.markdown("""
    <div style="
        background-color: #E7F1FF;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #0F52BA;
        margin-top: 20px;
    ">
        <h2 style="color: #0F52BA;">🎯 Nuestra propuesta de valor</h2>
        <p style="font-size: 1.05rem;">
        <b>Monitor Financiero & Cotizador de Créditos</b> transforma datos
        financieros complejos en información clara, visual y accionable,
        permitiendo comparar alternativas de crédito y apoyar decisiones
        financieras mediante análisis de datos y Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
# =============================================================================
# PESTAÑA 1: CARGAR Y EXPLORAR DATOS
# =============================================================================
with tab_eda:
    st.header("🔍 Carga de Archivo de Datos")
    st.markdown("Sube tu archivo `.csv` o `.xlsx` para mapear y procesar las variables automáticamente:")
    
    uploaded_file = st.file_uploader("Seleccionar archivo de datos", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            st.session_state.df_clean = clean_data(df_raw)
            st.success("✅ ¡Archivo cargado y columnas transformadas correctamente!")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

    st.markdown("---")
    
    if st.session_state.df_clean is not None:
        st.markdown("### 📋 Vista Previa de la Data Transformada")
        
        cols_mapeadas = [c for c in ['nombre_entidad', 'tipo_credito', 'tasa_efectiva_promedio', 'monto_desembolsado', 'numero_creditos', 'tipo_garantia', 'producto_credito', 'plazo_credito', 'tamano_empresa'] if c in st.session_state.df_clean.columns]
        st.info(f"Campos clave detectados y normalizados: **{', '.join(cols_mapeadas)}**")
        
        st.dataframe(st.session_state.df_clean, use_container_width=True)
    else:
        st.info("👆 Por favor sube un archivo CSV para generar el mapeo y habilitar las demás pestañas.")

# =============================================================================
# PESTAÑA 2: DASHBOARD DE TASAS Y MERCADO
# =============================================================================
with tab_dashboard:
    st.header("📊 Dashboard Financiero General")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ No se ha cargado ninguna data. Ve a la **Pestaña #1** y sube tu archivo CSV para ver los gráficos.")
    else:
        df_curr = st.session_state.df_clean
        
        col1, col2, col3 = st.columns(3)
        if 'tasa_efectiva_promedio' in df_curr.columns:
            col1.metric("Tasa Promedio Mercado", f"{df_curr['tasa_efectiva_promedio'].mean():.2f}% E.A.")
        if 'monto_desembolsado' in df_curr.columns:
            col2.metric("Volumen Desembolsado Total", f"${df_curr['monto_desembolsado'].sum()/1e6:,.0f} M")
        if 'numero_creditos' in df_curr.columns:
            col3.metric("Créditos Registrados", f"{df_curr['numero_creditos'].sum():,.0f}")
        
        st.markdown("---")
        
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            if 'tasa_efectiva_promedio' in df_curr.columns and 'nombre_entidad' in df_curr.columns:
                df_rank = df_curr.groupby('nombre_entidad')['tasa_efectiva_promedio'].mean().reset_index().sort_values(by='tasa_efectiva_promedio', ascending=True)
                
                fig_rank = px.bar(
                    df_rank,
                    x='tasa_efectiva_promedio',
                    y='nombre_entidad',
                    orientation='h',
                    title="Ranking de Tasas Efectivas Promedio (Menor a Mayor)",
                    color='tasa_efectiva_promedio',
                    color_continuous_scale='Blues_r',
                    template='plotly_white'
                )
                fig_rank.update_layout(
                    yaxis=dict(autorange="reversed"),
                    font=dict(color="#212529")
                )
                st.plotly_chart(fig_rank, use_container_width=True)
                
        with c_chart2:
            if 'tipo_credito' in df_curr.columns and 'monto_desembolsado' in df_curr.columns:
                fig_pie = px.pie(
                    df_curr,
                    names='tipo_credito',
                    values='monto_desembolsado',
                    hole=0.4,
                    title="Distribución del Crédito por Tipo",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    template='plotly_white'
                )
                fig_pie.update_layout(font=dict(color="#212529"))
                st.plotly_chart(fig_pie, use_container_width=True)

# =============================================================================
# PESTAÑA 3: CALCULADORA Y COMPARADOR DE CRÉDITOS
# =============================================================================
with tab_simulador:
    st.header("🧮 Simulación de Préstamo y Comparativa Bancaria")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ No se ha cargado ninguna data. Ve a la **Pestaña #1** y sube tu archivo CSV para simular créditos.")
    else:
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            monto_solicitado = st.number_input(
                "Monto a Solicitar ($ COP):", 
                min_value=500000, 
                max_value=1000000000, 
                value=10000000, 
                step=1000000
            )
        
        with col_in2:
            plazo_meses = st.slider(
                "Plazo (Meses):", 
                min_value=6, 
                max_value=120, 
                value=24, 
                step=6
            )
            
        with col_in3:
            if 'tipo_credito' in st.session_state.df_clean.columns:
                tipos_disponibles = list(st.session_state.df_clean['tipo_credito'].unique())
                tipo_credito_sel = st.selectbox("Tipo de Crédito:", tipos_disponibles)
            else:
                tipo_credito_sel = None

        st.markdown("---")
        
        if tipo_credito_sel and 'tipo_credito' in st.session_state.df_clean.columns:
            df_tasas = st.session_state.df_clean[st.session_state.df_clean['tipo_credito'] == tipo_credito_sel]
        else:
            df_tasas = st.session_state.df_clean

        if df_tasas.empty or 'tasa_efectiva_promedio' not in df_tasas.columns:
            st.warning("No hay datos de tasas válidos en el archivo cargado para realizar la simulación.")
        else:
            resumen_bancos = df_tasas.groupby('nombre_entidad')['tasa_efectiva_promedio'].mean().reset_index()
            
            resultados = []
            for _, row in resumen_bancos.iterrows():
                banco = row['nombre_entidad']
                tasa_ea = row['tasa_efectiva_promedio']
                cuota, intereses, total = calcular_cuota_fija(monto_solicitado, tasa_ea, plazo_meses)
                
                resultados.append({
                    'Entidad': banco,
                    'Tasa E.A. (%)': round(tasa_ea, 2),
                    'Tasa E.M. (%)': round(ea_to_em(tasa_ea) * 100, 2),
                    'Cuota Mensual ($)': cuota,
                    'Total Intereses ($)': intereses,
                    'Total a Pagar ($)': total
                })
                
            df_res = pd.DataFrame(resultados).sort_values(by='Cuota Mensual ($)')
            mejor = df_res.iloc[0]
            
            st.subheader("🏆 Resumen de la Mejor Opción")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🏆 Mejor Opción", mejor['Entidad'])
            m2.metric("📉 Menor Tasa E.A.", f"{mejor['Tasa E.A. (%)']}%")
            m3.metric("💰 Menor Cuota", f"${mejor['Cuota Mensual ($)']:,.0f}")
            m4.metric("💵 Total a Pagar", f"${mejor['Total a Pagar ($)']:,.0f}")
            
            st.markdown("### 📋 Cuadro Comparativo Completo por Banco")
            
            df_view = df_res.copy()
            df_view['Cuota Mensual ($)'] = df_view['Cuota Mensual ($)'].apply(lambda x: f"${x:,.0f}")
            df_view['Total Intereses ($)'] = df_view['Total Intereses ($)'].apply(lambda x: f"${x:,.0f}")
            df_view['Total a Pagar ($)'] = df_view['Total a Pagar ($)'].apply(lambda x: f"${x:,.0f}")
            df_view['Tasa E.A. (%)'] = df_view['Tasa E.A. (%)'].apply(lambda x: f"{x:.2f}%")
            df_view['Tasa E.M. (%)'] = df_view['Tasa E.M. (%)'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(df_view, use_container_width=True)
            
            fig_bar = px.bar(
                df_res,
                x='Entidad',
                y='Cuota Mensual ($)',
                color='Tasa E.A. (%)',
                title=f"Comparativa de Cuotas Mensuales para ${monto_solicitado:,.0f} a {plazo_meses} meses",
                text_auto=',.0f',
                color_continuous_scale='Teal_r',
                template='plotly_white'
            )
            fig_bar.update_layout(font=dict(color="#212529"))
            st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# PESTAÑA 4: ANÁLISIS E INTERPRETACIÓN
# =============================================================================
with tab_analysis:
    st.header("📈 Diagnóstico e Interpretación Financiera")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ No se ha cargado ninguna data. Ve a la **Pestaña #1** y sube tu archivo CSV para visualizar el diagnóstico.")
    else:
        st.markdown("Analizamos los datos cargados en la **Pestaña #2 (Dashboard)** para entregarte un diagnóstico claro y dinámico de la mejor opción del mercado.")
        
        df_curr = st.session_state.df_clean
        
        if df_curr.empty or 'tasa_efectiva_promedio' not in df_curr.columns or 'nombre_entidad' not in df_curr.columns:
            st.warning("Se requieren columnas válidas de 'entidad' y 'tasa' en el archivo cargado para generar el análisis interpretativo.")
        else:
            # Ordenamiento ascendente (de menor a mayor tasa)
            df_rank = df_curr.groupby('nombre_entidad')['tasa_efectiva_promedio'].mean().reset_index().sort_values(by='tasa_efectiva_promedio', ascending=True)
            
            mejor_banco = df_rank.iloc[0]
            peor_banco = df_rank.iloc[-1]
            tasa_promedio_mkt = df_rank['tasa_efectiva_promedio'].mean()
            diferencial_tasas = peor_banco['tasa_efectiva_promedio'] - mejor_banco['tasa_efectiva_promedio']
            ahorro_vs_promedio = tasa_promedio_mkt - mejor_banco['tasa_efectiva_promedio']

            st.markdown("### 📊 Datos Consolidados del Mercado")
            c1, c2 = st.columns([2, 1])
            
            with c1:
                fig_rank_tab4 = px.bar(
                    df_rank,
                    x='tasa_efectiva_promedio',
                    y='nombre_entidad',
                    orientation='h',
                    title="Ranking de Tasas Efectivas Promedio (Menor a Mayor)",
                    color='tasa_efectiva_promedio',
                    color_continuous_scale='Blues_r',
                    template='plotly_white'
                )
                fig_rank_tab4.update_layout(
                    yaxis=dict(autorange="reversed"),
                    font=dict(color="#212529")
                )
                st.plotly_chart(fig_rank_tab4, use_container_width=True, key="chart_rank_tab4")
                
            with c2:
                st.metric("Tasa Promedio Mercado", f"{tasa_promedio_mkt:.2f}% E.A.")
                st.metric("🥇 Mejor Entidad", f"{mejor_banco['nombre_entidad']}", f"{mejor_banco['tasa_efectiva_promedio']:.2f}% E.A.")
                st.metric("🔻 Entidad Más Costosa", f"{peor_banco['nombre_entidad']}", f"{peor_banco['tasa_efectiva_promedio']:.2f}% E.A.")
                st.metric("Spread de Mercado", f"{diferencial_tasas:.2f}% E.A.")

            st.markdown("---")
            
            st.markdown("### 🧠 ¿Qué significan estos números para tu dinero?")

            st.success(f"""
            ### 🏆 1. La Opción Ganadora: **{mejor_banco['nombre_entidad']}**
            * **Tasa Ofrecida:** **{mejor_banco['tasa_efectiva_promedio']:.2f}% E.A.**
            * **Ventaja Clave:** Se ubica **{ahorro_vs_promedio:.2f}% por debajo** del promedio del mercado. Es la alternativa que menor costo financiero generará sobre tu capital desembolsado.
            """)

            st.warning(f"""
            ### ⚠️ 2. La Opción Menos Conveniente: **{peor_banco['nombre_entidad']}**
            * **Tasa Ofrecida:** **{peor_banco['tasa_efectiva_promedio']:.2f}% E.A.**
            * **Impacto Financiero:** Hay una brecha de **{diferencial_tasas:.2f}%** entre la mejor y la peor opción. Tomar tu crédito aquí implica pagar intereses efectivamente más altos por exactamente la misma suma.
            """)

            st.info(f"""
            ### 💡 3. Guía Rápida para Decidir
            * **Criterio de Elección:** Busca siempre entidades cuyas tasas estén **por debajo de {tasa_promedio_mkt:.2f}% E.A.** (Promedio del Mercado).
            * **Siguiente Paso:** Ve a la **Pestaña #3 (Calculadora)**, ingresa el monto exacto que necesitas y valida cuánto te ahorras en la cuota mensual eligiendo a **{mejor_banco['nombre_entidad']}**.
            """)

# =============================================================================
# PESTAÑA 5: PREDICCIÓN CON MACHINE LEARNING (OPTIMIZADOR AUTOMÁTICO)
# =============================================================================
with tab_ml:
    st.header("🤖 Predicción Predictiva & Recomendador Automático de Tasas")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ No se ha cargado ninguna data. Ve a la **Pestaña #1** y sube tu archivo CSV para entrenar el modelo de Machine Learning.")
    else:
        st.markdown("""
        Ingresa el monto y el tipo de crédito deseado. El modelo predictivo de **Random Forest** evaluará internamente 
        todas las entidades financieras disponibles para determinar **cuál banco te ofrece la tasa estimada más baja** y conveniente.
        """)
        
        df_ml = st.session_state.df_clean.copy()
        
        req_cols = ['tasa_efectiva_promedio', 'monto_desembolsado', 'tipo_credito']
        if not all(col in df_ml.columns for col in req_cols):
            st.warning("El dataset necesita al menos las columnas `tasa_efectiva_promedio`, `monto_desembolsado` y `tipo_credito` para ejecutar la predicción.")
        else:
            # Preprocesamiento para ML
            feature_cols = ['monto_desembolsado', 'tipo_credito']
            has_banco = 'nombre_entidad' in df_ml.columns
            if has_banco:
                feature_cols.append('nombre_entidad')
                
            df_model_data = df_ml[feature_cols + ['tasa_efectiva_promedio']].dropna()
            
            if len(df_model_data) < 10:
                st.error("Se requieren al menos 10 registros válidos en el archivo para entrenar el modelo.")
            else:
                # One-Hot Encoding de variables categóricas
                cat_cols = [c for c in ['tipo_credito', 'nombre_entidad'] if c in df_model_data.columns]
                df_encoded = pd.get_dummies(df_model_data, columns=cat_cols, drop_first=False)
                
                X = df_encoded.drop(columns=['tasa_efectiva_promedio'])
                y = df_encoded['tasa_efectiva_promedio']
                
                # Entrenamiento del modelo
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X, y)
                
                st.success("✅ Modelo entrenado exitosamente.")
                
                st.markdown("---")
                st.subheader("🔮 Cotizador & Buscador de la Mejor Entidad")
                
                c_ml1, c_ml2, c_ml3 = st.columns(3)
                
                with c_ml1:
                    monto_pred = st.number_input(
                        "Monto a solicitar ($ COP):", 
                        min_value=1000000, 
                        max_value=500000000, 
                        value=20000000, 
                        step=1000000,
                        key="ml_monto_auto"
                    )
                    
                with c_ml2:
                    tipos_opt = list(df_ml['tipo_credito'].unique())
                    tipo_pred = st.selectbox("Tipo de Crédito:", tipos_opt, key="ml_tipo_auto")
                    
                with c_ml3:
                    plazo_pred = st.slider(
                        "Plazo estimado (Meses):", 
                        min_value=6, 
                        max_value=120, 
                        value=24, 
                        step=6,
                        key="ml_plazo_auto"
                    )
                        
                if st.button("🚀 Encontrar el Mejor Banco y Predecir Tasa", key="btn_ml_predict"):
                    list_bancos = df_ml['nombre_entidad'].unique() if has_banco else ['Mercado General']
                    
                    resultados_pred = []
                    
                    for banco in list_bancos:
                        # Vector de entrada en cero
                        input_row = pd.DataFrame(0, index=[0], columns=X.columns)
                        
                        if 'monto_desembolsado' in input_row.columns:
                            input_row['monto_desembolsado'] = monto_pred
                            
                        col_tipo = f"tipo_credito_{tipo_pred}"
                        if col_tipo in input_row.columns:
                            input_row[col_tipo] = 1
                            
                        if has_banco:
                            col_banco = f"nombre_entidad_{banco}"
                            if col_banco in input_row.columns:
                                input_row[col_banco] = 1
                                
                        tasa_est = rf_model.predict(input_row)[0]
                        cuota_est, int_est, total_est = calcular_cuota_fija(monto_pred, tasa_est, plazo_pred)
                        
                        resultados_pred.append({
                            'Entidad': banco,
                            'Tasa Estimada (E.A.)': tasa_est,
                            'Cuota Mensual ($)': cuota_est,
                            'Total Intereses ($)': int_est
                        })
                    
                    # Convertir a DataFrame y ordenar de MENOR a MAYOR tasa
                    df_preds = pd.DataFrame(resultados_pred).sort_values(by='Tasa Estimada (E.A.)', ascending=True)
                    mejor_opcion = df_preds.iloc[0]
                    
                    st.markdown("---")
                    st.success(f"### 🏆 Banco Sugerido: **{mejor_opcion['Entidad']}**")
                    
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("🥇 Entidad Recomendada", mejor_opcion['Entidad'])
                    p2.metric("📉 Tasa Estimada (E.A.)", f"{mejor_opcion['Tasa Estimada (E.A.)']:.2f}%")
                    p3.metric("💳 Cuota Mensual", f"${mejor_opcion['Cuota Mensual ($)']:,.0f}")
                    p4.metric("💰 Intereses Totales", f"${mejor_opcion['Total Intereses ($)']:,.0f}")
                    
                    st.markdown("### 📊 Comparativa de Tasas Estimadas por Entidad")
                    
                    # Gráfico ordenado de menor a mayor con azul predeterminado y KEY única
                    fig_ml_rank = px.bar(
                        df_preds,
                        x='Tasa Estimada (E.A.)',
                        y='Entidad',
                        orientation='h',
                        title=f"Ranking Predictivo para Crédito de {tipo_pred} por ${monto_pred:,.0f}",
                        color='Tasa Estimada (E.A.)',
                        color_continuous_scale='Blues_r',
                        template='plotly_white',
                        text_auto='.2f'
                    )
                    fig_ml_rank.update_layout(
                        yaxis=dict(autorange="reversed"),
                        font=dict(color="#212529")
                    )
                    st.plotly_chart(fig_ml_rank, use_container_width=True, key="chart_ml_rank_auto")
                    
                    # Tabla formateada
                    df_preds_view = df_preds.copy()
                    df_preds_view['Tasa Estimada (E.A.)'] = df_preds_view['Tasa Estimada (E.A.)'].apply(lambda x: f"{x:.2f}%")
                    df_preds_view['Cuota Mensual ($)'] = df_preds_view['Cuota Mensual ($)'].apply(lambda x: f"${x:,.0f}")
                    df_preds_view['Total Intereses ($)'] = df_preds_view['Total Intereses ($)'].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(df_preds_view, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y TEMA OSCURO
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Monitor Financiero & Cotizador de Créditos Colombia",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para forzar el tema oscuro
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E222A;
        border-radius: 4px;
        color: #E0E0E0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730 !important;
        color: #00FFB2 !important;
    }
    </style>
""", unsafe_allow_html_gradient=True)

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE CÁLCULO FINANCIERO (SISTEMA COLOMBIANO)
# -----------------------------------------------------------------------------
def ea_to_em(tasa_ea):
    """Convierte Tasa Efectiva Anual (E.A.) a Efectiva Mensual (E.M.)"""
    if pd.isna(tasa_ea) or tasa_ea <= 0:
        return 0.0
    return ((1 + tasa_ea / 100.0) ** (1.0 / 12.0) - 1.0)

def calcular_cuota_fija(monto, tasa_ea, plazo_meses):
    """Calcula cuota mensual fija (Amortización Sistema Francés)"""
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
    """Aplica reglas de negocio y limpieza a cualquier dataset cargado."""
    df = df.copy()

    # 1. Normalización de nombres de columnas
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

    # Mapeo flexible de columnas comunes
    col_map = {
        'nombre_de_la_entidad': 'nombre_entidad',
        'entidad': 'nombre_entidad',
        'tipo_de_credito': 'tipo_credito',
        'modalidad': 'tipo_credito',
        'tasa_efectiva_promedio_ponderada': 'tasa_efectiva_promedio',
        'tasa_ea': 'tasa_efectiva_promedio',
        'tasa': 'tasa_efectiva_promedio',
        'monto_desembolsado': 'monto_desembolsado',
        'monto': 'monto_desembolsado',
        'numero_de_creditos': 'numero_creditos',
        'creditos': 'numero_creditos'
    }
    df = df.rename(columns=col_map)

    # 2. Conversión de Fechas
    fecha_cols = [c for c in df.columns if 'fecha' in c or 'created' in c or 'updated' in c]
    for col in fecha_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3. Conversión de Tipos de Datos Numéricos
    num_cols = ['tasa_efectiva_promedio', 'monto_desembolsado', 'numero_creditos', 'margen_adicional_a_la']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Eliminación de Duplicados
    df = df.drop_duplicates()

    # 5. Tratamiento de Nulos
    if 'tasa_efectiva_promedio' in df.columns:
        df = df[df['tasa_efectiva_promedio'] > 0]
        df = df[(df['tasa_efectiva_promedio'] >= 1.0) & (df['tasa_efectiva_promedio'] <= 100.0)]
        
    if 'monto_desembolsado' in df.columns:
        df['monto_desembolsado'] = df['monto_desembolsado'].fillna(0)

    if 'tipo_credito' in df.columns:
        df['tipo_credito'] = df['tipo_credito'].fillna('General')

    if 'nombre_entidad' in df.columns:
        df['nombre_entidad'] = df['nombre_entidad'].fillna('Desconocido')

    return df

# Dataset base por defecto
@st.cache_data
def load_default_data():
    raw_data = [
        {"nombre_entidad": "BANCO DE BOGOTA", "tipo_credito": "Consumo", "tasa_efectiva_promedio": 18.50, "monto_desembolsado": 4500000000, "numero_creditos": 1200},
        {"nombre_entidad": "BANCOLOMBIA", "tipo_credito": "Consumo", "tasa_efectiva_promedio": 20.10, "monto_desembolsado": 8900000000, "numero_creditos": 3100},
        {"nombre_entidad": "DAVIVIENDA", "tipo_credito": "Consumo", "tasa_efectiva_promedio": 22.30, "monto_desembolsado": 6200000000, "numero_creditos": 2100},
        {"nombre_entidad": "BBVA COLOMBIA", "tipo_credito": "Consumo", "tasa_efectiva_promedio": 19.80, "monto_desembolsado": 3800000000, "numero_creditos": 980},
        {"nombre_entidad": "BANCO DE BOGOTA", "tipo_credito": "Vivienda", "tasa_efectiva_promedio": 14.20, "monto_desembolsado": 12000000000, "numero_creditos": 300},
        {"nombre_entidad": "BANCOLOMBIA", "tipo_credito": "Vivienda", "tasa_efectiva_promedio": 13.80, "monto_desembolsado": 18000000000, "numero_creditos": 520},
    ]
    return clean_data(pd.DataFrame(raw_data))

if 'df_clean' not in st.session_state:
    st.session_state.df_clean = load_default_data()

# -----------------------------------------------------------------------------
# INTERFAZ Y PESTAÑAS
# -----------------------------------------------------------------------------
st.title("🏦 Monitor Financiero & Cotizador de Créditos")

tab_simulador, tab_dashboard, tab_eda, tab_math = st.tabs([
    "🧮 1. Calculadora & Comparador", 
    "📊 2. Dashboard de Tasas", 
    "🔍 3. Cargar & Explorar Datos",
    "📐 4. Funciones Auxiliares Financieras"
])

# =============================================================================
# PESTAÑA 1: CALCULADORA Y COMPARADOR DE CRÉDITOS
# =============================================================================
with tab_simulador:
    st.header("🧮 Simulación de Préstamo y Comparativa Bancaria")
    
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
        st.warning("No hay información suficiente sobre tasas en el dataset actual.")
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
            template='plotly_dark'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# PESTAÑA 2: DASHBOARD DE TASAS Y MERCADO
# =============================================================================
with tab_dashboard:
    st.header("📊 Dashboard Financiero General")
    
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
            df_rank = df_curr.groupby('nombre_entidad')['tasa_efectiva_promedio'].mean().reset_index().sort_values(by='tasa_efectiva_promedio')
            fig_rank = px.bar(
                df_rank,
                x='tasa_efectiva_promedio',
                y='nombre_entidad',
                orientation='h',
                title="Ranking de Tasas Efectivas Promedio",
                color='tasa_efectiva_promedio',
                template='plotly_dark'
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
                template='plotly_dark'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# =============================================================================
# PESTAÑA 3: CARGAR Y EXPLORAR DATOS (EDA CUSTOM)
# =============================================================================
with tab_eda:
    st.header("🔍 Carga tu Archivo de Datos")
    st.markdown("Sube tu archivo en formato `.csv` o `.xlsx` para actualizar automáticamente las simulaciones:")
    
    uploaded_file = st.file_uploader("Selecciona un archivo", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_custom = pd.read_csv(uploaded_file)
            else:
                df_custom = pd.read_excel(uploaded_file)
            
            st.session_state.df_clean = clean_data(df_custom)
            st.success("✅ Archivo cargado y procesado exitosamente con el pipeline EDA.")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

    st.markdown("### 📋 Vista Previa de Datos Limpios Procesados")
    st.dataframe(st.session_state.df_clean, use_container_width=True)

# =============================================================================
# PESTAÑA 4: FUNCIONES AUXILIARES FINANCIERAS (DOCUMENTACIÓN MATEMÁTICA)
# =============================================================================
with tab_math:
    st.header("📐 Funciones Auxiliares de Cálculo Financiero")
    st.markdown("Esta sección describe la lógica matemática utilizada para simular las condiciones de crédito en Colombia:")
    
    st.subheader("1. Conversión de Tasa Efectiva Anual (E.A.) a Efectiva Mensual (E.M.)")
    st.markdown("Formula de equivalencia de tasas periódicas vencer de periodo vencido:")
    st.latex(r"i_m = (1 + i_{ea})^{\frac{1}{12}} - 1")
    
    st.subheader("2. Cálculo de Cuota Fija Mensual (Sistema Francés)")
    st.markdown("Fórmula utilizada para determinar el valor de la cuota fija mensual constante $C$ dada una tasa mensual $i_m$, un monto $P$ y un número de meses $n$:")
    st.latex(r"C = P \times \frac{i_m (1 + i_m)^n}{(1 + i_m)^n - 1}")

    st.subheader("3. Código Python Integrado")
    st.code("""
def ea_to_em(tasa_ea):
    return ((1 + tasa_ea / 100.0) ** (1.0 / 12.0) - 1.0)

def calcular_cuota_fija(monto, tasa_ea, plazo_meses):
    i_m = ea_to_em(tasa_ea)
    if i_m == 0:
        cuota = monto / plazo_meses
    else:
        cuota = monto * (i_m * ((1 + i_m) ** plazo_meses)) / (((1 + i_m) ** plazo_meses) - 1)
    
    total_pagar = cuota * plazo_meses
    total_intereses = total_pagar - monto
    return cuota, total_intereses, total_pagar
    """, language="python")

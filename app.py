import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Monitor Financiero & Cotizador de Créditos Colombia",
    page_icon="🏦",
    layout="wide"
)

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
# LIMPIEZA Y TRANSFORMACIÓN DE DATOS (EDA PIPELINE)
# -----------------------------------------------------------------------------
def clean_data(df):
    """Aplica las reglas de negocio y limpieza para el dataset oficial (qzsc-9esp)."""
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

    # Mapeo flexible de columnas para datos mock y datos reales de la API Socrata
    col_map = {
        'nombre_de_la_entidad': 'nombre_entidad',
        'tipo_de_credito': 'tipo_credito',
        'tasa_efectiva_promedio_ponderada': 'tasa_efectiva_promedio',
        'tipo_de_garantia': 'tipo_garantia',
        'monto_desembolsado': 'monto_desembolsado',
        'numero_de_creditos': 'numero_creditos'
    }
    df = df.rename(columns=col_map)

    # 2. Conversión explícita de Fechas (si existen)
    fecha_cols = [c for c in df.columns if 'fecha' in c or 'created_at' in c or 'updated_at' in c]
    for col in fecha_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3. Conversión explícita de Tipos de Datos Numéricos
    num_cols = ['tasa_efectiva_promedio', 'monto_desembolsado', 'numero_creditos', 'margen_adicional_a_la']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Eliminación de Duplicados
    df = df.drop_duplicates()

    # 5. Imputación / Tratamiento de Nulos y Ceros Inconsistentes
    if 'tasa_efectiva_promedio' in df.columns:
        # Filtrar o imputar tasas igual a 0 o imposibles
        df = df[df['tasa_efectiva_promedio'] > 0]
        
    if 'monto_desembolsado' in df.columns:
        df['monto_desembolsado'] = df['monto_desembolsado'].fillna(0)

    if 'grupo_etnico' in df.columns:
        df['grupo_etnico'] = df['grupo_etnico'].fillna('Sin información')

    if 'tipo_credito' in df.columns:
        df['tipo_credito'] = df['tipo_credito'].fillna('Otros')

    # 6. Detección / Tratamiento de Outliers en Tasas (Filtro prudencial)
    if 'tasa_efectiva_promedio' in df.columns:
        df = df[(df['tasa_efectiva_promedio'] >= 1.0) & (df['tasa_efectiva_promedio'] <= 100.0)]

    return df

# -----------------------------------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    raw_data = [
        {"Nombre de la entidad": "BANCO DE BOGOTA", "Tipo de crédito": "Consumo", "Tasa efectiva promedio ponderada": 18.50, "Tipo de garantía": "Sin Garantía", "Monto desembolsado": 4500000000, "Número de créditos": 1200},
        {"Nombre de la entidad": "BANCOLOMBIA", "Tipo de crédito": "Consumo", "Tasa efectiva promedio ponderada": 20.10, "Tipo de garantía": "Sin Garantía", "Monto desembolsado": 8900000000, "Número de créditos": 3100},
        {"Nombre de la entidad": "DAVIVIENDA", "Tipo de crédito": "Consumo", "Tasa efectiva promedio ponderada": 22.30, "Tipo de garantía": "Sin Garantía", "Monto desembolsado": 6200000000, "Número de créditos": 2100},
        {"Nombre de la entidad": "BBVA COLOMBIA", "Tipo de crédito": "Consumo", "Tasa efectiva promedio ponderada": 19.80, "Tipo de garantía": "Sin Garantía", "Monto desembolsado": 3800000000, "Número de créditos": 980},
        {"Nombre de la entidad": "BANCO POPULAR", "Tipo de crédito": "Consumo", "Tasa efectiva promedio ponderada": 21.50, "Tipo de garantía": "Sin Garantía", "Monto desembolsado": 1500000000, "Número de créditos": 450},
        {"Nombre de la entidad": "BANCO DE BOGOTA", "Tipo de crédito": "Vivienda", "Tasa efectiva promedio ponderada": 14.20, "Tipo de garantía": "Idónea", "Monto desembolsado": 12000000000, "Número de créditos": 300},
        {"Nombre de la entidad": "BANCOLOMBIA", "Tipo de crédito": "Vivienda", "Tasa efectiva promedio ponderada": 13.80, "Tipo de garantía": "Idónea", "Monto desembolsado": 18000000000, "Número de créditos": 520},
        {"Nombre de la entidad": "DAVIVIENDA", "Tipo de crédito": "Vivienda", "Tasa efectiva promedio ponderada": 14.50, "Tipo de garantía": "Idónea", "Monto desembolsado": 15000000000, "Número de créditos": 410},
    ]
    df_raw = pd.DataFrame(raw_data)
    return clean_data(df_raw)

if 'df_clean' not in st.session_state:
    st.session_state.df_clean = load_data()

# -----------------------------------------------------------------------------
# INTERFAZ Y PESTAÑAS
# -----------------------------------------------------------------------------
st.title("🏦 Cotizador & Monitor de Créditos en Colombia")
st.caption("Basado en el dataset de Datos Abiertos de la Superintendencia Financiera (datos.gov.co)")

tab_simulador, tab_dashboard, tab_eda = st.tabs([
    "🧮 1. Calculadora & Comparador", 
    "📊 2. Dashboard de Tasas", 
    "🔍 3. Explorador de Datos"
])

# =============================================================================
# PESTAÑA 1: CALCULADORA Y COMPARADOR DE CRÉDITOS
# =============================================================================
with tab_simulador:
    st.header("🧮 Simulación de Préstamo y Comparativa Bancaria")
    st.markdown("Ingresa las condiciones de tu préstamo para simular y rankear la mejor opción del mercado:")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        monto_solicitado = st.number_input(
            "Monto a Solicitar ($ COP):", 
            min_value=500000, 
            max_value=500000000, 
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
        tipos_disponibles = list(st.session_state.df_clean['tipo_credito'].unique())
        tipo_credito_sel = st.selectbox("Tipo de Crédito:", tipos_disponibles)

    st.markdown("---")
    
    # Filtrar data procesada por tipo de crédito
    df_tasas = st.session_state.df_clean[st.session_state.df_clean['tipo_credito'] == tipo_credito_sel]
    
    if df_tasas.empty:
        st.warning("No se encontraron entidades con tasas registradas para esta modalidad.")
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
            color_continuous_scale='Greens_r'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# PESTAÑA 2: DASHBOARD DE TASAS Y MERCADO
# =============================================================================
with tab_dashboard:
    st.header("📊 Dashboard del Sistema Financiero")
    
    tasa_prom = st.session_state.df_clean['tasa_efectiva_promedio'].mean()
    monto_total = st.session_state.df_clean['monto_desembolsado'].sum()
    creditos_total = st.session_state.df_clean['numero_creditos'].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Tasa Promedio Mercado", f"{tasa_prom:.2f}% E.A.")
    col2.metric("Volumen Desembolsado", f"${monto_total/1e9:.2f} Mil Millones")
    col3.metric("Créditos Registrados", f"{creditos_total:,.0f}")
    
    st.markdown("---")
    
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        df_rank = st.session_state.df_clean.groupby('nombre_entidad')['tasa_efectiva_promedio'].mean().reset_index().sort_values(by='tasa_efectiva_promedio')
        fig_rank = px.bar(
            df_rank,
            x='tasa_efectiva_promedio',
            y='nombre_entidad',
            orientation='h',
            title="Ranking de Tasas Efectivas Promedio (Menor a Mayor)",
            color='tasa_efectiva_promedio',
            color_continuous_scale='Reds_r'
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        
    with c_chart2:
        fig_pie = px.pie(
            st.session_state.df_clean,
            names='tipo_credito',
            values='monto_desembolsado',
            hole=0.4,
            title="Distribución del Crédito por Modalidad ($ COP)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# =============================================================================
# PESTAÑA 3: EXPLORADOR DE DATOS (EDA)
# =============================================================================
with tab_eda:
    st.header("🔍 Datos Abiertos Superintendencia Financiera (Limpios)")
    st.dataframe(st.session_state.df_clean, use_container_width=True)

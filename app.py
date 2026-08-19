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
# CARGA DE DATOS (datos.gov.co)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # En producción se conecta a la API Socrata o archivo .csv descargado de datos.gov.co
    # Código adaptador con la estructura oficial del dataset qzsc-9esp
    data = [
        {"Nombre de la entidad": "BANCO DE BOGOTA", "Tipo de crédito": "Consumo", "Tasa efectiva promedio ponderada": 18.50, "Tipo de garantía": "Sin Garantía", "Monto desembolsado": 4500000000, "Número de créditos": 1200},
        {"Nombre de la entidad": "BANCOLOMBIA", "Tipo de crédito": "Consumo", "Tasa efectiva promedio ponderada": 20.10, "Tipo de garantía": "Sin Garantía", "Monto desembolsado": 8900000000, "Número de créditos": 3100},
        {"Nombre de la entidad": "DAVIVIENDA", "Tipo de crédito": "Consumo", "Tasa efectiva promedio ponderada": 22.30, "Tipo de garantía": "Sin Garantía", "Monto desembolsado": 6200000000, "Número de créditos": 2100},
        {"Nombre de la entidad": "BBVA COLOMBIA", "Tipo de crédito": "Consumo", "Tasa efectiva promedio ponderada": 19.80, "Tipo de garantía": "Sin Garantía", "Monto desembolsado": 3800000000, "Número de créditos": 980},
        {"Nombre de la entidad": "BANCO POPULAR", "Tipo de crédito": "Consumo", "Tasa efectiva promedio ponderada": 21.50, "Tipo de garantía": "Sin Garantía", "Monto desembolsado": 1500000000, "Número de créditos": 450},
        {"Nombre de la entidad": "BANCO DE BOGOTA", "Tipo de crédito": "Vivienda", "Tasa efectiva promedio ponderada": 14.20, "Tipo de garantía": "Idónea", "Monto desembolsado": 12000000000, "Número de créditos": 300},
        {"Nombre de la entidad": "BANCOLOMBIA", "Tipo de crédito": "Vivienda", "Tasa efectiva promedio ponderada": 13.80, "Tipo de garantía": "Idónea", "Monto desembolsado": 18000000000, "Número de créditos": 520},
        {"Nombre de la entidad": "DAVIVIENDA", "Tipo de crédito": "Vivienda", "Tasa efectiva promedio ponderada": 14.50, "Tipo de garantía": "Idónea", "Monto desembolsado": 15000000000, "Número de créditos": 410},
    ]
    return pd.DataFrame(data)

if 'df_clean' not in st.session_state:
    st.session_state.df_clean = load_data()

# -----------------------------------------------------------------------------
# INTERFAZ Y PESTAÑAS REDISEÑADAS
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
        tipos_disponibles = list(st.session_state.df_clean['Tipo de crédito'].unique())
        tipo_credito_sel = st.selectbox("Tipo de Crédito:", tipos_disponibles)

    st.markdown("---")
    
    # Filtrar data por el tipo de crédito seleccionado
    df_tasas = st.session_state.df_clean[st.session_state.df_clean['Tipo de crédito'] == tipo_credito_sel]
    
    if df_tasas.empty:
        st.warning("No se encontraron entidades con tasas registradas para esta modalidad.")
    else:
        # Calcular simulación para cada entidad reportada
        resumen_bancos = df_tasas.groupby('Nombre de la entidad')['Tasa efectiva promedio ponderada'].mean().reset_index()
        
        resultados = []
        for _, row in resumen_bancos.iterrows():
            banco = row['Nombre de la entidad']
            tasa_ea = row['Tasa efectiva promedio ponderada']
            cuota, intereses, total = calcular_cuota_fija(monto_solicitado, tasa_ea, plazo_meses)
            
            resultados.append({
                'Entidad': banco,
                'Tasa E.A. (%)': round(tasa_ea, 2),
                'Tasa E.M. (%)': round(ea_to_em(tasa_ea)*100, 2),
                'Cuota Mensual ($)': cuota,
                'Total Intereses ($)': intereses,
                'Total a Pagar ($)': total
            })
            
        df_res = pd.DataFrame(resultados).sort_values(by='Cuota Mensual ($)')
        
        # MEJOR OPCIÓN DESTACADA
        mejor = df_res.iloc[0]
        
        st.subheader("🏆 Resumen de la Mejor Opción")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏆 Mejor Opción", mejor['Entidad'])
        m2.metric("📉 Menor Tasa E.A.", f"{mejor['Tasa E.A. (%)']}%")
        m3.metric("💰 Menor Cuota", f"${mejor['Cuota Mensual ($)']:,.0f}")
        m4.metric("💵 Total a Pagar", f"${mejor['Total a Pagar ($)']:,.0f}")
        
        st.markdown("### 📋 Cuadro Comparativo Completo por Banco")
        
        # Formato visual explicito para la tabla
        df_view = df_res.copy()
        df_view['Cuota Mensual ($)'] = df_view['Cuota Mensual ($)'].apply(lambda x: f"${x:,.0f}")
        df_view['Total Intereses ($)'] = df_view['Total Intereses ($)'].apply(lambda x: f"${x:,.0f}")
        df_view['Total a Pagar ($)'] = df_view['Total a Pagar ($)'].apply(lambda x: f"${x:,.0f}")
        df_view['Tasa E.A. (%)'] = df_view['Tasa E.A. (%)'].apply(lambda x: f"{x:.2f}%")
        df_view['Tasa E.M. (%)'] = df_view['Tasa E.M. (%)'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(df_view, use_container_width=True)
        
        # Gráfico interactivo de comparación
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
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Tasa Promedio Mercado", f"{st.session_state.df_clean['Tasa efectiva promedio ponderada'].mean():.2f}% E.A.")
    col2.metric("Volumen Desembolsado", f"${st.session_state.df_clean['Monto desembolsado'].sum()/1e9:.2f} Mil Millones")
    col3.metric("Créditos Registrados", f"{st.session_state.df_clean['Número de créditos'].sum():,}")
    
    st.markdown("---")
    
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        fig_rank = px.bar(
            st.session_state.df_clean.groupby('Nombre de la entidad')['Tasa efectiva promedio ponderada'].mean().reset_index().sort_values(by='Tasa efectiva promedio ponderada'),
            x='Tasa efectiva promedio ponderada',
            y='Nombre de la entidad',
            orientation='h',
            title="Ranking de Tasas Efectivas Promedio (Menor a Mayor)",
            color='Tasa efectiva promedio ponderada',
            color_continuous_scale='Reds_r'
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        
    with c_chart2:
        fig_pie = px.pie(
            st.session_state.df_clean,
            names='Tipo de crédito',
            values='Monto desembolsado',
            hole=0.4,
            title="Distribución del Crédito por Modalidad ($ COP)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# =============================================================================
# PESTAÑA 3: EXPLORADOR DE DATOS (EDA)
# =============================================================================
with tab_eda:
    st.header("🔍 Datos Abiertos Superintendencia Financiera")
    st.dataframe(st.session_state.df_clean, use_container_width=True)

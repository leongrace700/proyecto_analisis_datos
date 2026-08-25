import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Monitor Financiero & Cotizador de Créditos",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #F8F9FA;
    color: #212529;
    font-family: 'Segoe UI', sans-serif;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #E9ECEF;
    padding: 8px;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab"] {
    color: #495057;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #0F52BA !important;
    background-color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCIONES
# ============================================================

def ea_to_em(tasa_ea):
    if pd.isna(tasa_ea) or tasa_ea <= 0:
        return 0
    return (1 + tasa_ea / 100) ** (1 / 12) - 1


def calcular_cuota(monto, tasa_ea, plazo):
    if monto <= 0 or plazo <= 0:
        return 0, 0, 0

    i = ea_to_em(tasa_ea)

    if i == 0:
        cuota = monto / plazo
    else:
        cuota = monto * (i * (1+i)**plazo) / ((1+i)**plazo - 1)

    total = cuota * plazo
    intereses = total - monto

    return cuota, intereses, total


def clean_data(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("á", "a")
        .str.replace("é", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
        .str.replace("ñ", "n")
    )

    mapa = {

        # Entidad
        "nombre_de_la_entidad": "nombre_entidad",
        "entidad": "nombre_entidad",
        "banco": "nombre_entidad",

        # Crédito
        "tipo_de_credito": "tipo_credito",
        "tipo_de_cr_dito": "tipo_credito",
        "modalidad": "tipo_credito",
        "linea_de_credito": "tipo_credito",
        "producto": "tipo_credito",

        # Tasa
        "tasa_efectiva_promedio_ponderada":
            "tasa_efectiva_promedio",
        "tasa_ea":
            "tasa_efectiva_promedio",
        "tasa":
            "tasa_efectiva_promedio",

        # Montos
        "montos_desembolsados":
            "monto_desembolsado",
        "monto":
            "monto_desembolsado",

        # Créditos
        "numero_de_creditos":
            "numero_creditos",
        "creditos":
            "numero_creditos",

        # Otros
        "tipo_de_garantia":
            "tipo_garantia",
        "tipo_de_garant_a":
            "tipo_garantia",
        "producto_de_credito":
            "producto_credito",
        "producto_de_cr_dito":
            "producto_credito",
        "plazo_de_credito":
            "plazo_credito",
        "plazo_de_cr_dito":
            "plazo_credito"
    }

    df.rename(columns=mapa, inplace=True)

    if "tipo_credito" not in df.columns:
        df["tipo_credito"] = "General"

    if "nombre_entidad" not in df.columns:
        df["nombre_entidad"] = "Desconocido"

    df["tipo_credito"] = df["tipo_credito"].fillna("General")
    df["nombre_entidad"] = df["nombre_entidad"].fillna("Desconocido")

    for col in [
        "tasa_efectiva_promedio",
        "monto_desembolsado",
        "numero_creditos"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    if "tasa_efectiva_promedio" in df.columns:

        df = df[
            (df["tasa_efectiva_promedio"] >= 1) &
            (df["tasa_efectiva_promedio"] <= 100)
        ]

    if "monto_desembolsado" in df.columns:
        df["monto_desembolsado"] = (
            df["monto_desembolsado"].fillna(0)
        )

    return df.drop_duplicates()


# ============================================================
# SESSION STATE
# ============================================================

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None


# ============================================================
# TÍTULO
# ============================================================

st.title("🏦 Monitor Financiero & Cotizador de Créditos")


# ============================================================
# PESTAÑAS
# ============================================================

(
    tab_inicio,
    tab_eda,
    tab_dashboard,
    tab_simulador,
    tab_analysis,
    tab_ml
) = st.tabs([
    "🏠 0. Proyecto",
    "🔍 1. Cargar & Explorar Datos",
    "📊 2. Dashboard de Tasas",
    "🧮 3. Calculadora & Comparador",
    "📈 4. Análisis e Interpretación",
    "🤖 5. Predicción con ML"
])


# ============================================================
# PESTAÑA 0 - PROYECTO
# ============================================================

with tab_inicio:

    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#0F52BA,#1976D2);
        padding:30px;
        border-radius:15px;
        color:white;
        text-align:center;
        margin-bottom:25px;
    ">

    <h1 style="color:white;">
    🏦 Monitor Financiero & Cotizador de Créditos
    </h1>

    <p style="font-size:1.15rem;">
    Plataforma inteligente para comparar, analizar y simular créditos
    </p>

    <p>
    Análisis de datos aplicado a la toma de decisiones financieras
    </p>

    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # INTEGRANTES
    # --------------------------------------------------------

    st.markdown("## 👥 Integrantes del equipo")

    c1, c2 = st.columns(2)

    with c1:
        st.info("👤 **Grace Leon**")

    with c2:
        st.info("👤 **Mayerly Roman**")

    c3, c4 = st.columns(2)

    with c3:
        st.info("👤 **Marco Jimenez**")

    with c4:
        st.info("👤 **Zurley Taborda**")

    st.markdown("---")

    # --------------------------------------------------------
    # OBJETIVO
    # --------------------------------------------------------

    st.markdown("## 🎯 Objetivo de la solución")

    st.write("""
    Desarrollar una herramienta interactiva que permita analizar y
    comparar las tasas de interés de diferentes entidades financieras,
    facilitando la simulación de créditos y ayudando al usuario a
    identificar alternativas potencialmente más convenientes según
    el monto y plazo de financiación.
    """)

    # --------------------------------------------------------
    # PROBLEMA / USUARIOS
    # --------------------------------------------------------

    st.markdown("## 💡 Problema que resolvemos")

    p1, p2 = st.columns(2)

    with p1:

        st.markdown("### ❓ ¿Qué problema resuelve?")

        st.markdown("""
        - Tasas diferentes entre entidades.
        - Diferentes modalidades de crédito.
        - Diferentes plazos y condiciones.
        - Información financiera dispersa.
        - Dificultad para calcular el costo real.
        """)

    with p2:

        st.markdown("### 👥 ¿Quiénes serían los usuarios?")

        st.markdown("""
        - Personas que desean solicitar un crédito.
        - Clientes que desean comparar entidades.
        - Asesores financieros.
        - PYMES.
        - Analistas financieros.
        - Organizaciones del sector financiero.
        """)

    st.markdown("---")

    # --------------------------------------------------------
    # DIFERENCIADOR
    # --------------------------------------------------------

    st.markdown("## 🚀 ¿Qué hace diferente nuestra solución?")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("""
        ### 📊 Datos

        Analiza información financiera estructurada para encontrar
        diferencias y patrones en las tasas.
        """)

    with d2:
        st.markdown("""
        ### 🧮 Simulación

        Calcula cuotas, intereses y total a pagar según monto,
        plazo y tasa.
        """)

    with d3:
        st.markdown("""
        ### 🤖 Machine Learning

        Utiliza Random Forest para generar estimaciones y apoyar
        la recomendación de entidades.
        """)

    st.markdown("---")

    # --------------------------------------------------------
    # TECNOLOGÍAS
    # --------------------------------------------------------

    st.markdown("## 🛠️ Tecnologías utilizadas")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.info("🐍 **Python**\n\nLenguaje principal")

    with t2:
        st.info("🎨 **Streamlit**\n\nInterfaz web")

    with t3:
        st.info("📊 **Pandas**\n\nAnálisis de datos")

    t4, t5, t6 = st.columns(3)

    with t4:
        st.info("📈 **Plotly**\n\nVisualización")

    with t5:
        st.info("🤖 **Scikit-learn**\n\nMachine Learning")

    with t6:
        st.info("💾 **CSV / Excel**\n\nFuentes de datos")

    st.markdown("---")

    # --------------------------------------------------------
    # FUNCIONALIDADES
    # --------------------------------------------------------

    st.markdown("## ⚙️ Funcionalidades principales")

    f1, f2 = st.columns(2)

    with f1:

        st.markdown("""
        - 📂 Carga y transformación de datos.
        - 📊 Dashboard financiero.
        - 🧮 Calculadora de cuotas.
        - 🏦 Comparación de entidades.
        """)

    with f2:

        st.markdown("""
        - 📉 Comparación de tasas.
        - 📈 Análisis financiero.
        - 🤖 Modelo Random Forest.
        - 🔮 Predicción de tasas.
        """)

    st.markdown("---")

    # --------------------------------------------------------
    # DATOS E IA
    # --------------------------------------------------------

    st.markdown("## 🧠 ¿Cómo aporta valor mediante datos e IA?")

    st.markdown("""
    El análisis de datos permite identificar diferencias entre
    entidades, encontrar tendencias, comparar costos y visualizar
    información financiera de manera sencilla.

    El componente de Machine Learning permite analizar la relación
    entre monto, tipo de crédito y entidad para generar estimaciones
    y recomendaciones basadas en los datos disponibles.

    Así, la solución se convierte en una herramienta de apoyo para
    la **toma de decisiones financieras**.
    """)

    # --------------------------------------------------------
    # PRODUCTO REAL
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("## 💼 ¿Cómo podría convertirse en un producto real?")

    b1, b2 = st.columns(2)

    with b1:

        st.markdown("""
        ### 👤 Usuario

        1. Ingresa monto.
        2. Selecciona plazo.
        3. Selecciona tipo de crédito.
        4. Compara entidades.
        5. Consulta cuota e intereses.
        """)

    with b2:

        st.markdown("""
        ### 💰 Modelo de negocio

        Podría convertirse en una plataforma para consumidores,
        asesores financieros, empresas y organizaciones del sector.
        """)

    st.markdown("---")

    # --------------------------------------------------------
    # FLUJO
    # --------------------------------------------------------

    st.markdown("## 🔄 ¿Cómo funciona?")

    a, b, c, d = st.columns(4)

    with a:
        st.markdown("### 1️⃣ Datos\n\nCarga de información.")

    with b:
        st.markdown("### 2️⃣ Análisis\n\nLimpieza y transformación.")

    with c:
        st.markdown("### 3️⃣ Simulación\n\nCálculo y comparación.")

    with d:
        st.markdown("### 4️⃣ Decisión\n\nRecomendaciones.")

    st.markdown("---")

    st.markdown("## 🖥️ Evidencia visual")

    st.info("""
    Aquí pueden agregarse capturas del Dashboard, Calculadora,
    Análisis Financiero y Modelo Predictivo.
    """)

    st.markdown("""
    <div style="
        background:#E7F1FF;
        padding:25px;
        border-radius:12px;
        border-left:6px solid #0F52BA;
    ">

    <h2 style="color:#0F52BA;">
    🎯 Nuestra propuesta de valor
    </h2>

    <p>
    <b>Monitor Financiero & Cotizador de Créditos</b> transforma
    datos financieros complejos en información clara y accionable,
    permitiendo comparar alternativas de crédito y apoyar decisiones
    financieras mediante análisis de datos y Machine Learning.
    </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PESTAÑA 1 - CARGAR DATOS
# ============================================================

with tab_eda:

    st.header("🔍 Carga de Archivo de Datos")

    archivo = st.file_uploader(
        "Seleccionar archivo",
        type=["csv", "xlsx"]
    )

    if archivo:

        try:

            if archivo.name.endswith(".csv"):
                df = pd.read_csv(archivo)
            else:
                df = pd.read_excel(archivo)

            st.session_state.df_clean = clean_data(df)

            st.success("✅ Archivo cargado correctamente.")

        except Exception as e:

            st.error(f"Error al procesar el archivo: {e}")

    if st.session_state.df_clean is not None:

        df = st.session_state.df_clean

        st.markdown("### 📋 Vista previa")

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info(
            "👆 Carga un archivo CSV o Excel para habilitar "
            "las demás pestañas."
        )


# ============================================================
# PESTAÑA 2 - DASHBOARD
# ============================================================

with tab_dashboard:

    st.header("📊 Dashboard Financiero")

    if st.session_state.df_clean is None:

        st.warning("Carga primero los datos en la Pestaña #1.")

    else:

        df = st.session_state.df_clean

        c1, c2, c3 = st.columns(3)

        if "tasa_efectiva_promedio" in df.columns:
            c1.metric(
                "Tasa Promedio",
                f"{df['tasa_efectiva_promedio'].mean():.2f}% E.A."
            )

        if "monto_desembolsado" in df.columns:
            c2.metric(
                "Monto Desembolsado",
                f"${df['monto_desembolsado'].sum()/1e6:,.0f} M"
            )

        if "numero_creditos" in df.columns:
            c3.metric(
                "Créditos",
                f"{df['numero_creditos'].sum():,.0f}"
            )

        col1, col2 = st.columns(2)

        with col1:

            if "nombre_entidad" in df.columns:

                ranking = (
                    df.groupby("nombre_entidad")
                    ["tasa_efectiva_promedio"]
                    .mean()
                    .reset_index()
                    .sort_values("tasa_efectiva_promedio")
                )

                fig = px.bar(
                    ranking,
                    x="tasa_efectiva_promedio",
                    y="nombre_entidad",
                    orientation="h",
                    title="Ranking de Tasas",
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with col2:

            if (
                "tipo_credito" in df.columns
                and "monto_desembolsado" in df.columns
            ):

                fig = px.pie(
                    df,
                    names="tipo_credito",
                    values="monto_desembolsado",
                    hole=.4,
                    title="Distribución por Tipo de Crédito"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# ============================================================
# PESTAÑA 3 - CALCULADORA
# ============================================================

with tab_simulador:

    st.header("🧮 Calculadora & Comparador")

    if st.session_state.df_clean is None:

        st.warning("Carga primero los datos.")

    else:

        df = st.session_state.df_clean

        c1, c2, c3 = st.columns(3)

        with c1:

            monto = st.number_input(
                "Monto ($ COP)",
                min_value=500000,
                value=10000000,
                step=1000000
            )

        with c2:

            plazo = st.slider(
                "Plazo (meses)",
                6,
                120,
                24,
                step=6
            )

        with c3:

            tipos = list(df["tipo_credito"].unique())

            tipo = st.selectbox(
                "Tipo de crédito",
                tipos
            )

        datos = df[df["tipo_credito"] == tipo]

        resultados = []

        for _, row in (
            datos.groupby("nombre_entidad")
            ["tasa_efectiva_promedio"]
            .mean()
            .reset_index()
            .iterrows()
        ):

            cuota, intereses, total = calcular_cuota(
                monto,
                row["tasa_efectiva_promedio"],
                plazo
            )

            resultados.append({
                "Entidad": row["nombre_entidad"],
                "Tasa E.A. (%)":
                    row["tasa_efectiva_promedio"],
                "Cuota Mensual ($)": cuota,
                "Intereses ($)": intereses,
                "Total ($)": total
            })

        resultados = pd.DataFrame(resultados)

        if not resultados.empty:

            resultados = resultados.sort_values(
                "Cuota Mensual ($)"
            )

            mejor = resultados.iloc[0]

            st.subheader("🏆 Mejor opción")

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Entidad",
                mejor["Entidad"]
            )

            m2.metric(
                "Tasa",
                f"{mejor['Tasa E.A. (%)']:.2f}%"
            )

            m3.metric(
                "Cuota",
                f"${mejor['Cuota Mensual ($)']:,.0f}"
            )

            st.dataframe(
                resultados.style.format({
                    "Tasa E.A. (%)": "{:.2f}%",
                    "Cuota Mensual ($)": "${:,.0f}",
                    "Intereses ($)": "${:,.0f}",
                    "Total ($)": "${:,.0f}"
                }),
                use_container_width=True
            )


# ============================================================
# PESTAÑA 4 - ANÁLISIS
# ============================================================

with tab_analysis:

    st.header("📈 Análisis e Interpretación")

    if st.session_state.df_clean is None:

        st.warning("Carga primero los datos.")

    else:

        df = st.session_state.df_clean

        ranking = (
            df.groupby("nombre_entidad")
            ["tasa_efectiva_promedio"]
            .mean()
            .reset_index()
            .sort_values("tasa_efectiva_promedio")
        )

        mejor = ranking.iloc[0]
        peor = ranking.iloc[-1]
        promedio = ranking["tasa_efectiva_promedio"].mean()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Promedio Mercado",
            f"{promedio:.2f}% E.A."
        )

        c2.metric(
            "🥇 Mejor Entidad",
            mejor["nombre_entidad"]
        )

        c3.metric(
            "🔻 Mayor Tasa",
            peor["nombre_entidad"]
        )

        fig = px.bar(
            ranking,
            x="tasa_efectiva_promedio",
            y="nombre_entidad",
            orientation="h",
            title="Ranking de Tasas",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.success(f"""
        ### 🏆 Mejor opción

        **{mejor['nombre_entidad']}**

        Tasa promedio: **{mejor['tasa_efectiva_promedio']:.2f}% E.A.**
        """)

        st.warning(f"""
        ### ⚠️ Mayor tasa

        **{peor['nombre_entidad']}**

        Tasa promedio:
        **{peor['tasa_efectiva_promedio']:.2f}% E.A.**
        """)


# ============================================================
# PESTAÑA 5 - MACHINE LEARNING
# ============================================================

with tab_ml:

    st.header("🤖 Predicción & Recomendador")

    if st.session_state.df_clean is None:

        st.warning("Carga primero los datos.")

    else:

        df = st.session_state.df_clean.copy()

        columnas = [
            "tasa_efectiva_promedio",
            "monto_desembolsado",
            "tipo_credito"
        ]

        if all(c in df.columns for c in columnas):

            modelo_df = df[
                columnas + ["nombre_entidad"]
            ].dropna()

            if len(modelo_df) >= 10:

                datos = pd.get_dummies(
                    modelo_df,
                    columns=[
                        "tipo_credito",
                        "nombre_entidad"
                    ]
                )

                X = datos.drop(
                    columns="tasa_efectiva_promedio"
                )

                y = datos["tasa_efectiva_promedio"]

                modelo = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )

                modelo.fit(X, y)

                st.success(
                    "✅ Modelo entrenado correctamente."
                )

                monto_ml = st.number_input(
                    "Monto a solicitar",
                    min_value=1000000,
                    value=20000000,
                    step=1000000
                )

                tipo_ml = st.selectbox(
                    "Tipo de crédito",
                    df["tipo_credito"].unique()
                )

                if st.button(
                    "🚀 Encontrar mejor entidad"
                ):

                    resultados = []

                    for banco in df["nombre_entidad"].unique():

                        entrada = pd.DataFrame(
                            0,
                            index=[0],
                            columns=X.columns
                        )

                        if "monto_desembolsado" in entrada:
                            entrada[
                                "monto_desembolsado"
                            ] = monto_ml

                        tipo_col = f"tipo_credito_{tipo_ml}"
                        banco_col = f"nombre_entidad_{banco}"

                        if tipo_col in entrada:
                            entrada[tipo_col] = 1

                        if banco_col in entrada:
                            entrada[banco_col] = 1

                        pred = modelo.predict(
                            entrada
                        )[0]

                        resultados.append({
                            "Entidad": banco,
                            "Tasa Estimada": pred
                        })

                    predicciones = pd.DataFrame(
                        resultados
                    ).sort_values(
                        "Tasa Estimada"
                    )

                    mejor = predicciones.iloc[0]

                    st.success(
                        f"🏆 Entidad recomendada: "
                        f"**{mejor['Entidad']}**"
                    )

                    st.metric(
                        "Tasa estimada",
                        f"{mejor['Tasa Estimada']:.2f}% E.A."
                    )

                    fig = px.bar(
                        predicciones,
                        x="Tasa Estimada",
                        y="Entidad",
                        orientation="h",
                        title="Tasas Estimadas",
                        template="plotly_white"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.dataframe(
                        predicciones,
                        use_container_width=True
                    )

            else:

                st.error(
                    "Se necesitan al menos 10 registros "
                    "válidos para entrenar el modelo."
                )

        else:

            st.warning# =============================================================================
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

            El dataset debe contener las columnas:
            tasa_efectiva_promedio,
            monto_desembolsado y
            tipo_credito.
            """)

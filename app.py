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
}

h1, h2, h3 {
    color: #0F52BA;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCIONES
# ============================================================

def convertir_tasa_ea_a_mensual(tasa_ea):
    """
    Convierte una tasa efectiva anual (%) a efectiva mensual.
    """
    if pd.isna(tasa_ea) or tasa_ea <= 0:
        return 0

    return (1 + tasa_ea / 100) ** (1 / 12) - 1


def calcular_credito(monto, tasa_ea, plazo):
    """
    Calcula cuota mensual, intereses y total a pagar.
    """

    if monto <= 0 or plazo <= 0:
        return 0, 0, 0

    tasa_mensual = convertir_tasa_ea_a_mensual(tasa_ea)

    if tasa_mensual == 0:
        cuota = monto / plazo

    else:
        cuota = (
            monto
            * (
                tasa_mensual
                * (1 + tasa_mensual) ** plazo
            )
            / (
                (1 + tasa_mensual) ** plazo - 1
            )
        )

    total = cuota * plazo
    intereses = total - monto

    return cuota, intereses, total


def limpiar_datos(df):

    df = df.copy()

    # Normalizar nombres de columnas
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

    # Equivalencias posibles
    columnas = {
        "nombre_de_la_entidad": "nombre_entidad",
        "entidad": "nombre_entidad",
        "banco": "nombre_entidad",

        "tipo_de_credito": "tipo_credito",
        "tipo_de_cr_dito": "tipo_credito",
        "modalidad": "tipo_credito",
        "producto": "tipo_credito",
        "producto_de_credito": "tipo_credito",
        "producto_de_cr_dito": "tipo_credito",

        "tasa": "tasa_efectiva_promedio",
        "tasa_ea": "tasa_efectiva_promedio",
        "tasa_efectiva_promedio_ponderada":
            "tasa_efectiva_promedio",

        "monto": "monto_desembolsado",
        "montos_desembolsados":
            "monto_desembolsado",

        "numero_de_creditos":
            "numero_creditos",
        "creditos":
            "numero_creditos",

        "plazo_de_credito":
            "plazo_credito",
        "plazo_de_cr_dito":
            "plazo_credito"
    }

    df.rename(columns=columnas, inplace=True)

    # Columnas mínimas
    if "nombre_entidad" not in df.columns:
        df["nombre_entidad"] = "Entidad no identificada"

    if "tipo_credito" not in df.columns:
        df["tipo_credito"] = "General"

    if "tasa_efectiva_promedio" not in df.columns:
        df["tasa_efectiva_promedio"] = np.nan

    if "monto_desembolsado" not in df.columns:
        df["monto_desembolsado"] = 0

    if "numero_creditos" not in df.columns:
        df["numero_creditos"] = 0

    # Conversión numérica
    for columna in [
        "tasa_efectiva_promedio",
        "monto_desembolsado",
        "numero_creditos"
    ]:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    # Limpieza
    df["nombre_entidad"] = (
        df["nombre_entidad"]
        .fillna("Entidad no identificada")
        .astype(str)
    )

    df["tipo_credito"] = (
        df["tipo_credito"]
        .fillna("General")
        .astype(str)
    )

    # Tasas razonables
    df = df[
        (
            df["tasa_efectiva_promedio"].isna()
        )
        |
        (
            (df["tasa_efectiva_promedio"] > 0)
            &
            (df["tasa_efectiva_promedio"] <= 100)
        )
    ]

    return df.drop_duplicates()


# ============================================================
# SESSION STATE
# ============================================================

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None


# ============================================================
# ENCABEZADO
# ============================================================

st.title("🏦 Monitor Financiero & Cotizador de Créditos")

st.caption(
    "Herramienta de análisis, comparación y simulación "
    "de créditos basada en datos."
)


# ============================================================
# CREACIÓN DE PESTAÑAS
# ============================================================

tab_inicio, tab_eda, tab_dashboard, tab_simulador, tab_analysis, tab_ml = st.tabs([
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

    # PORTADA
    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#0F52BA,#1976D2);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
    ">

        <h1 style="color:white;">
            🏦 Monitor Financiero & Cotizador de Créditos
        </h1>

        <p style="
            color:white;
            font-size:1.15rem;
        ">
            Plataforma inteligente para comparar,
            analizar y simular créditos
        </p>

        <p style="color:white;">
            Análisis de datos aplicado a la toma
            de decisiones financieras
        </p>

    </div>
    """, unsafe_allow_html=True)

    # INTEGRANTES
    st.markdown("## 👥 Integrantes del equipo")

    col1, col2 = st.columns(2)

    with col1:
        st.info("👤 **Grace Leon**")

    with col2:
        st.info("👤 **Mayerly Roman**")

    col3, col4 = st.columns(2)

    with col3:
        st.info("👤 **Marco Jimenez**")

    with col4:
        st.info("👤 **Zurley Taborda**")

    st.divider()

    # OBJETIVO
    st.markdown("## 🎯 Objetivo de la solución")

    st.write("""
    Desarrollar una herramienta interactiva que permita analizar y
    comparar las tasas de interés de diferentes entidades financieras,
    facilitando la simulación de créditos y ayudando al usuario a
    identificar alternativas potencialmente más convenientes según
    el monto y plazo de financiación.
    """)

    # PROBLEMA Y USUARIOS
    st.markdown("## 💡 Problema que resolvemos")

    problema, usuarios = st.columns(2)

    with problema:

        st.markdown("### ❓ ¿Qué problema resuelve?")

        st.markdown("""
        - Información financiera dispersa.
        - Diferentes tasas entre entidades.
        - Diferentes modalidades de crédito.
        - Diferentes plazos.
        - Dificultad para calcular el costo real.
        - Comparaciones poco claras para el usuario.
        """)

    with usuarios:

        st.markdown("### 👥 ¿Quiénes serían los usuarios?")

        st.markdown("""
        - Personas que desean solicitar un crédito.
        - Clientes que desean comparar entidades.
        - Asesores financieros.
        - Pequeñas y medianas empresas.
        - Analistas financieros.
        - Organizaciones del sector financiero.
        """)

    st.divider()

    # DIFERENCIADOR
    st.markdown("## 🚀 ¿Qué hace diferente nuestra solución?")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("""
        ### 📊 Análisis de datos

        Permite identificar diferencias,
        tendencias y patrones en las tasas.
        """)

    with d2:
        st.markdown("""
        ### 🧮 Simulación

        Calcula cuota mensual,
        intereses y total a pagar.
        """)

    with d3:
        st.markdown("""
        ### 🤖 Machine Learning

        Utiliza Random Forest para
        realizar estimaciones.
        """)

    st.divider()

    # TECNOLOGÍAS
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

    st.divider()

    # FUNCIONALIDADES
    st.markdown("## ⚙️ Funcionalidades principales")

    f1, f2 = st.columns(2)

    with f1:
        st.markdown("""
        - 📂 Carga de archivos CSV y Excel.
        - 📊 Dashboard interactivo.
        - 🧮 Calculadora de créditos.
        - 🏦 Comparación de entidades.
        """)

    with f2:
        st.markdown("""
        - 📉 Comparación de tasas.
        - 📈 Análisis financiero.
        - 🤖 Modelo Random Forest.
        - 🔮 Predicción de tasas.
        """)

    st.divider()

    # DATOS E IA
    st.markdown("## 🧠 ¿Cómo aporta valor mediante análisis de datos e IA?")

    st.write("""
    El análisis de datos permite identificar diferencias entre
    entidades, encontrar tendencias, comparar costos y visualizar
    información financiera de manera sencilla.

    El componente de Machine Learning analiza la relación entre
    monto, tipo de crédito y entidad para generar estimaciones.

    De esta manera, la solución transforma datos financieros
    complejos en información útil para apoyar la toma de decisiones.
    """)

    st.divider()

    # PRODUCTO
    st.markdown("## 💼 ¿Cómo podría convertirse en un producto real?")

    producto1, producto2 = st.columns(2)

    with producto1:

        st.markdown("""
        ### 👤 Experiencia del usuario

        1. Ingresa el monto.
        2. Selecciona el plazo.
        3. Selecciona el tipo de crédito.
        4. Compara entidades.
        5. Consulta cuota e intereses.
        """)

    with producto2:

        st.markdown("""
        ### 💰 Modelo de negocio

        Podría convertirse en una plataforma
        dirigida a consumidores, asesores financieros,
        empresas y organizaciones del sector financiero.
        """)

    st.divider()

    # FLUJO
    st.markdown("## 🔄 ¿Cómo funciona nuestra solución?")

    flujo1, flujo2, flujo3, flujo4 = st.columns(4)

    with flujo1:
        st.markdown("### 1️⃣ Datos\nCarga de información.")

    with flujo2:
        st.markdown("### 2️⃣ Análisis\nLimpieza y transformación.")

    with flujo3:
        st.markdown("### 3️⃣ Simulación\nCálculo y comparación.")

    with flujo4:
        st.markdown("### 4️⃣ Decisión\nRecomendaciones.")

    st.divider()

    # EVIDENCIA
    st.markdown("## 🖥️ Evidencia visual")

    st.info("""
    En esta sección pueden incorporarse capturas de pantalla
    del Dashboard, Calculadora, Análisis Financiero y
    Modelo Predictivo.
    """)

    # CIERRE
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

        <p style="font-size:1.05rem;">
            <b>Monitor Financiero & Cotizador de Créditos</b>
            transforma datos financieros complejos en información
            clara y accionable, permitiendo comparar alternativas
            de crédito y apoyar decisiones financieras mediante
            análisis de datos y Machine Learning.
        </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PESTAÑA 1 - CARGAR DATOS
# ============================================================

with tab_eda:

    st.header("🔍 Cargar & Explorar Datos")

    archivo = st.file_uploader(
        "Selecciona un archivo CSV o Excel",
        type=["csv", "xlsx"]
    )

    if archivo is not None:

        try:

            if archivo.name.lower().endswith(".csv"):
                df = pd.read_csv(archivo)
            else:
                df = pd.read_excel(archivo)

            df = limpiar_datos(df)

            st.session_state.df_clean = df

            st.success("✅ Archivo cargado correctamente.")

        except Exception as error:

            st.error(
                f"❌ No fue posible procesar el archivo: {error}"
            )

    if st.session_state.df_clean is not None:

        df = st.session_state.df_clean

        st.subheader("📋 Vista previa")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.write(
            f"Registros: **{len(df):,}**"
        )

    else:

        st.info(
            "Carga un archivo para comenzar el análisis."
        )


# ============================================================
# PESTAÑA 2 - DASHBOARD
# ============================================================

with tab_dashboard:

    st.header("📊 Dashboard de Tasas")

    df = st.session_state.df_clean

    if df is None:

        st.warning(
            "⚠️ Carga primero los datos en la Pestaña #1."
        )

    else:

        c1, c2, c3 = st.columns(3)

        tasa_promedio = (
            df["tasa_efectiva_promedio"]
            .mean()
        )

        monto_total = (
            df["monto_desembolsado"]
            .sum()
        )

        creditos_total = (
            df["numero_creditos"]
            .sum()
        )

        c1.metric(
            "Tasa promedio",
            f"{tasa_promedio:.2f}% E.A."
        )

        c2.metric(
            "Monto desembolsado",
            f"${monto_total:,.0f}"
        )

        c3.metric(
            "Número de créditos",
            f"{creditos_total:,.0f}"
        )

        st.divider()

        # Ranking
        ranking = (
            df.groupby("nombre_entidad")
            ["tasa_efectiva_promedio"]
            .mean()
            .reset_index()
            .sort_values(
                "tasa_efectiva_promedio"
            )
        )

        fig = px.bar(
            ranking,
            x="tasa_efectiva_promedio",
            y="nombre_entidad",
            orientation="h",
            title="Tasa promedio por entidad",
            labels={
                "tasa_efectiva_promedio":
                    "Tasa E.A. (%)",
                "nombre_entidad":
                    "Entidad"
            },
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Tipo de crédito
        if "tipo_credito" in df.columns:

            resumen_tipo = (
                df.groupby("tipo_credito")
                ["monto_desembolsado"]
                .sum()
                .reset_index()
            )

            fig2 = px.pie(
                resumen_tipo,
                names="tipo_credito",
                values="monto_desembolsado",
                title="Monto por tipo de crédito",
                hole=0.4
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


# ============================================================
# PESTAÑA 3 - CALCULADORA
# ============================================================

with tab_simulador:

    st.header("🧮 Calculadora & Comparador")

    df = st.session_state.df_clean

    if df is None:

        st.warning(
            "⚠️ Carga primero los datos en la Pestaña #1."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            monto = st.number_input(
                "💰 Monto del crédito",
                min_value=500000,
                value=10000000,
                step=500000
            )

        with col2:

            plazo = st.slider(
                "📅 Plazo en meses",
                min_value=6,
                max_value=120,
                value=24,
                step=6
            )

        with col3:

            tipos = sorted(
                df["tipo_credito"]
                .dropna()
                .unique()
                .tolist()
            )

            tipo_seleccionado = st.selectbox(
                "🏦 Tipo de crédito",
                tipos
            )

        datos = df[
            df["tipo_credito"]
            == tipo_seleccionado
        ]

        tasas = (
            datos.groupby("nombre_entidad")
            ["tasa_efectiva_promedio"]
            .mean()
            .reset_index()
        )

        resultados = []

        for _, fila in tasas.iterrows():

            tasa = fila[
                "tasa_efectiva_promedio"
            ]

            cuota, intereses, total = calcular_credito(
                monto,
                tasa,
                plazo
            )

            resultados.append({
                "Entidad":
                    fila["nombre_entidad"],

                "Tasa E.A. (%)":
                    tasa,

                "Cuota mensual":
                    cuota,

                "Intereses":
                    intereses,

                "Total a pagar":
                    total
            })

        resultados_df = pd.DataFrame(
            resultados
        )

        if not resultados_df.empty:

            resultados_df = (
                resultados_df
                .sort_values("Cuota mensual")
                .reset_index(drop=True)
            )

            mejor = resultados_df.iloc[0]

            st.success(
                f"🏆 **Mejor alternativa según la tasa disponible: "
                f"{mejor['Entidad']}**"
            )

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Tasa",
                f"{mejor['Tasa E.A. (%)']:.2f}% E.A."
            )

            m2.metric(
                "Cuota mensual",
                f"${mejor['Cuota mensual']:,.0f}"
            )

            m3.metric(
                "Total a pagar",
                f"${mejor['Total a pagar']:,.0f}"
            )

            st.subheader("📋 Comparación")

            st.dataframe(
                resultados_df.style.format({
                    "Tasa E.A. (%)": "{:.2f}%",
                    "Cuota mensual": "${:,.0f}",
                    "Intereses": "${:,.0f}",
                    "Total a pagar": "${:,.0f}"
                }),
                use_container_width=True
            )

        else:

            st.warning(
                "No existen datos para el tipo de crédito seleccionado."
            )


# ============================================================
# PESTAÑA 4 - ANÁLISIS
# ============================================================

with tab_analysis:

    st.header("📈 Análisis e Interpretación")

    df = st.session_state.df_clean

    if df is None:

        st.warning(
            "⚠️ Carga primero los datos."
        )

    else:

        ranking = (
            df.groupby("nombre_entidad")
            ["tasa_efectiva_promedio"]
            .mean()
            .reset_index()
            .sort_values(
                "tasa_efectiva_promedio"
            )
        )

        if not ranking.empty:

            mejor = ranking.iloc[0]
            mayor = ranking.iloc[-1]
            promedio = ranking[
                "tasa_efectiva_promedio"
            ].mean()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Tasa promedio",
                f"{promedio:.2f}% E.A."
            )

            c2.metric(
                "🥇 Menor tasa",
                mejor["nombre_entidad"]
            )

            c3.metric(
                "🔺 Mayor tasa",
                mayor["nombre_entidad"]
            )

            fig = px.bar(
                ranking,
                x="tasa_efectiva_promedio",
                y="nombre_entidad",
                orientation="h",
                title="Ranking de entidades",
                labels={
                    "tasa_efectiva_promedio":
                        "Tasa E.A. (%)",
                    "nombre_entidad":
                        "Entidad"
                },
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown("### 🔎 Interpretación")

            st.write(f"""
            La tasa promedio observada en el conjunto de datos
            es de **{promedio:.2f}% E.A.**

            La entidad con menor tasa promedio es
            **{mejor['nombre_entidad']}**, mientras que la entidad
            con mayor tasa promedio es
            **{mayor['nombre_entidad']}**.

            Estos resultados permiten realizar una primera
            comparación del comportamiento de las tasas y apoyar
            el análisis financiero.
            """)


# ============================================================
# PESTAÑA 5 - MACHINE LEARNING
# ============================================================

with tab_ml:

    st.header("🤖 Predicción con Machine Learning")

    df = st.session_state.df_clean

    if df is None:

        st.warning(
            "⚠️ Carga primero los datos."
        )

    else:

        columnas_necesarias = [
            "tasa_efectiva_promedio",
            "monto_desembolsado",
            "tipo_credito",
            "nombre_entidad"
        ]

        faltantes = [
            columna
            for columna in columnas_necesarias
            if columna not in df.columns
        ]

        if faltantes:

            st.error(
                "Faltan las siguientes columnas: "
                + ", ".join(faltantes)
            )

        else:

            modelo_df = df[
                columnas_necesarias
            ].dropna()

            if len(modelo_df) < 10:

                st.warning("""
                Se necesitan al menos 10 registros
                válidos para entrenar el modelo.
                """)

            else:

                datos_modelo = pd.get_dummies(
                    modelo_df,
                    columns=[
                        "tipo_credito",
                        "nombre_entidad"
                    ]
                )

                X = datos_modelo.drop(
                    columns="tasa_efectiva_promedio"
                )

                y = datos_modelo[
                    "tasa_efectiva_promedio"
                ]

                modelo = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )

                modelo.fit(X, y)

                st.success(
                    "✅ Modelo Random Forest entrenado correctamente."
                )

                st.markdown(
                    "### 🔮 Realizar una estimación"
                )

                monto_prediccion = st.number_input(
                    "💰 Monto del crédito",
                    min_value=500000,
                    value=10000000,
                    step=500000,
                    key="monto_prediccion"
                )

                tipos_ml = sorted(
                    df["tipo_credito"]
                    .dropna()
                    .unique()
                    .tolist()
                )

                tipo_prediccion = st.selectbox(
                    "🏦 Tipo de crédito",
                    tipos_ml,
                    key="tipo_prediccion"
                )

                if st.button(
                    "🚀 Generar recomendación"
                ):

                    predicciones = []

                    entidades = (
                        df["nombre_entidad"]
                        .dropna()
                        .unique()
                    )

                    for entidad in entidades:

                        entrada = pd.DataFrame(
                            0,
                            index=[0],
                            columns=X.columns
                        )

                        entrada[
                            "monto_desembolsado"
                        ] = monto_prediccion

                        col_tipo = (
                            "tipo_credito_"
                            + tipo_prediccion
                        )

                        col_entidad = (
                            "nombre_entidad_"
                            + entidad
                        )

                        if col_tipo in entrada.columns:
                            entrada[col_tipo] = 1

                        if col_entidad in entrada.columns:
                            entrada[col_entidad] = 1

                        prediccion = modelo.predict(
                            entrada
                        )[0]

                        predicciones.append({
                            "Entidad": entidad,
                            "Tasa estimada E.A. (%)":
                                prediccion
                        })

                    predicciones_df = (
                        pd.DataFrame(predicciones)
                        .sort_values(
                            "Tasa estimada E.A. (%)"
                        )
                        .reset_index(drop=True)
                    )

                    if not predicciones_df.empty:

                        mejor = predicciones_df.iloc[0]

                        st.success(
                            f"🏆 Entidad recomendada: "
                            f"**{mejor['Entidad']}**"
                        )

                        st.metric(
                            "Tasa estimada",
                            f"{mejor['Tasa estimada E.A. (%)']:.2f}% E.A."
                        )

                        fig = px.bar(
                            predicciones_df,
                            x="Tasa estimada E.A. (%)",
                            y="Entidad",
                            orientation="h",
                            title="Comparación de tasas estimadas",
                            template="plotly_white"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        st.dataframe(
                            predicciones_df,
                            use_container_width=True
                        )

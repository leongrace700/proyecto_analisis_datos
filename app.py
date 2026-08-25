# =============================================================================
# PESTAÑA 0: PRESENTACIÓN DEL PROYECTO / PITCH
# =============================================================================
with tab_inicio:

    # -------------------------------------------------------------------------
    # PORTADA
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0F52BA 0%, #1976D2 100%);
        padding: 32px 25px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        text-align: center;
    ">
        <h1 style="
            color: white;
            margin-bottom: 10px;
            font-size: 2.1rem;
        ">
            🏦 Monitor Financiero & Cotizador de Créditos
        </h1>

        <p style="
            font-size: 1.15rem;
            margin-bottom: 8px;
        ">
            Plataforma inteligente para comparar, analizar y simular créditos
        </p>

        <p style="
            font-size: 0.95rem;
            margin-bottom: 0px;
        ">
            Análisis de datos aplicado a la toma de decisiones financieras
        </p>
    </div>
    """, unsafe_allow_html=True)


    # -------------------------------------------------------------------------
    # INTEGRANTES
    # -------------------------------------------------------------------------
    st.markdown("## 👥 Integrantes del equipo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="
            background-color: #E7F1FF;
            border: 1px solid #D5E5FA;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin-bottom: 12px;
        ">
            <p style="
                color: #0F52BA;
                font-size: 1.05rem;
                font-weight: 600;
                margin: 0;
            ">
                👤 Grace Leon
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background-color: #E7F1FF;
            border: 1px solid #D5E5FA;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin-bottom: 12px;
        ">
            <p style="
                color: #0F52BA;
                font-size: 1.05rem;
                font-weight: 600;
                margin: 0;
            ">
                👤 Mayerly Roman
            </p>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div style="
            background-color: #E7F1FF;
            border: 1px solid #D5E5FA;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin-bottom: 12px;
        ">
            <p style="
                color: #0F52BA;
                font-size: 1.05rem;
                font-weight: 600;
                margin: 0;
            ">
                👤 Marco Jimenez
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="
            background-color: #E7F1FF;
            border: 1px solid #D5E5FA;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin-bottom: 12px;
        ">
            <p style="
                color: #0F52BA;
                font-size: 1.05rem;
                font-weight: 600;
                margin: 0;
            ">
                👤 Zurley Taborda
            </p>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")


    # -------------------------------------------------------------------------
    # OBJETIVO
    # -------------------------------------------------------------------------
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


    # -------------------------------------------------------------------------
    # PROBLEMA Y USUARIOS
    # -------------------------------------------------------------------------
    st.markdown("## 💡 Problema que resolvemos")

    problema_col, usuario_col = st.columns(2)

    with problema_col:

        st.markdown("### ❓ ¿Qué problema resuelve?")

        st.markdown("""
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

        st.markdown("### 👥 ¿Quiénes serían los usuarios?")

        st.markdown("""
        La solución puede estar dirigida a:

        - Personas que desean solicitar un crédito.
        - Clientes que quieren comparar diferentes entidades.
        - Asesores financieros.
        - Pequeñas y medianas empresas.
        - Analistas del sector financiero.
        - Organizaciones que analizan el mercado crediticio.
        """)


    st.markdown("---")


    # -------------------------------------------------------------------------
    # DIFERENCIADOR
    # -------------------------------------------------------------------------
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


    # -------------------------------------------------------------------------
    # TECNOLOGÍAS
    # -------------------------------------------------------------------------
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
                min-height: 105px;
                text-align: center;
            ">
                <h3 style="
                    color: #0F52BA;
                    margin-bottom: 8px;
                ">
                    {icono} {nombre}
                </h3>

                <p style="
                    margin: 0;
                    font-size: 0.9rem;
                ">
                    {descripcion}
                </p>
            </div>
            """, unsafe_allow_html=True)


    # -------------------------------------------------------------------------
    # FUNCIONALIDADES
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## ⚙️ Funcionalidades principales")

    funcionalidades = [
        "📂 Carga y transformación automática de archivos CSV y Excel.",
        "📊 Dashboard interactivo para analizar tasas y comportamiento del mercado.",
        "🧮 Calculadora de cuotas para diferentes montos y plazos.",
        "🏦 Comparación de entidades financieras.",
        "📉 Identificación de tasas más bajas.",
        "📈 Análisis e interpretación de indicadores.",
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


    # -------------------------------------------------------------------------
    # VALOR MEDIANTE DATOS E IA
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🧠 ¿Cómo aporta valor mediante análisis de datos e IA?")

    st.markdown("""
    La plataforma convierte información financiera en indicadores y
    recomendaciones que facilitan la toma de decisiones.

    **El análisis de datos permite:**

    - Identificar diferencias entre las tasas de las entidades.
    - Encontrar tendencias y patrones.
    - Comparar el costo potencial de diferentes alternativas.
    - Visualizar información financiera de forma sencilla.

    **El componente de Machine Learning permite:**

    - Analizar la relación entre monto, tipo de crédito y entidad.
    - Estimar posibles tasas de interés.
    - Generar recomendaciones basadas en los datos disponibles.

    De esta manera, la solución pasa de ser solamente un dashboard a
    convertirse en una herramienta de **apoyo para la toma de decisiones**.
    """)


    # -------------------------------------------------------------------------
    # PRODUCTO / SERVICIO
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 💼 ¿Cómo podría convertirse en un producto real?")

    st.markdown("""
    La solución podría evolucionar hacia una plataforma web o aplicación
    financiera donde los usuarios ingresen sus necesidades de financiación
    y reciban una comparación personalizada.
    """)

    producto1, producto2 = st.columns(2)

    with producto1:

        st.markdown("""
        ### 👤 Para el usuario

        - Ingresa monto.
        - Selecciona plazo.
        - Selecciona tipo de crédito.
        - Compara entidades.
        - Consulta cuota e intereses.
        """)

    with producto2:

        st.markdown("""
        ### 💼 Modelo comercial

        Podría ofrecerse como servicio para:

        - Consumidores.
        - Asesores financieros.
        - Empresas.
        - Entidades del sector financiero.
        - Plataformas de comparación financiera.
        """)


    # -------------------------------------------------------------------------
    # FLUJO DE LA SOLUCIÓN
    # -------------------------------------------------------------------------
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

        Limpieza y transformación.
        """)

    with flujo3:
        st.markdown("""
        ### 3️⃣ Simulación

        Cálculo y comparación.
        """)

    with flujo4:
        st.markdown("""
        ### 4️⃣ Decisión

        Recomendaciones basadas en datos.
        """)


    # -------------------------------------------------------------------------
    # EVIDENCIA VISUAL
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🖥️ Evidencia visual de la solución")

    st.info("""
    📌 En esta sección se pueden incorporar capturas de pantalla de:

    - Dashboard de tasas.
    - Calculadora y comparador.
    - Análisis financiero.
    - Modelo predictivo.
    """)


    # -------------------------------------------------------------------------
    # CIERRE / PROPUESTA DE VALOR
    # -------------------------------------------------------------------------
    st.markdown("---")

    st.markdown("""
    <div style="
        background-color: #E7F1FF;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #0F52BA;
        margin-top: 20px;
    ">

        <h2 style="
            color: #0F52BA;
            margin-bottom: 12px;
        ">
            🎯 Nuestra propuesta de valor
        </h2>

        <p style="
            font-size: 1.05rem;
            line-height: 1.6;
            margin-bottom: 0;
        ">
            <b>Monitor Financiero & Cotizador de Créditos</b> transforma
            datos financieros complejos en información clara, visual y
            accionable, permitiendo comparar alternativas de crédito y
            apoyar decisiones financieras mediante análisis de datos y
            Machine Learning.
        </p>

    </div>
    """, unsafe_allow_html=True)

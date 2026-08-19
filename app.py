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

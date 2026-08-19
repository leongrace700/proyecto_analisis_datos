import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from lightgbm import LGBMClassifier

# TensorFlow es opcional: la app funciona sin Deep Learning si no está instalado.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="Credit Risk Platform",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Plataforma Integral de Riesgo Crediticio")
st.caption("EDA → Dashboard → Comparación ML → Deep Learning → Simulador → XAI")


# ============================================================
# SESSION STATE
# ============================================================
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

if "models" not in st.session_state:
    st.session_state.models = {}

if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None

if "model_features" not in st.session_state:
    st.session_state.model_features = []


# ============================================================
# FUNCIONES
# ============================================================
def clean_data(df, drop_dups=True, impute=True, clean_age=True):
    data = df.copy()

    if drop_dups:
        data = data.drop_duplicates()

    if impute:
        for col in data.columns:
            if data[col].isnull().any():
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].fillna(data[col].median())
                else:
                    mode = data[col].mode()
                    if len(mode):
                        data[col] = data[col].fillna(mode.iloc[0])

    if clean_age and "person_age" in data.columns:
        data = data[data["person_age"] <= 100]

    if "person_age" in data.columns:
        data["person_age"] = data["person_age"].astype(int)

    if "loan_status" in data.columns:
        data["loan_status"] = data["loan_status"].astype(int)

    return data


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

    return preprocessor


def metrics_row(name, y_true, pred, prob):
    return {
        "Modelo": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1-Score": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob)
    }


# ============================================================
# TABS
# ============================================================
tab_eda, tab_dashboard, tab_models, tab_simulator, tab_xai = st.tabs([
    "📁 1. EDA & Limpieza",
    "📊 2. Dashboard",
    "🤖 3. Modelos ML + DL",
    "💰 4. Simulador",
    "⚡ 5. XAI con SHAP"
])


# ============================================================
# 1. EDA
# ============================================================
with tab_eda:
    st.header("🔍 Carga y limpieza de datos")

    uploaded_file = st.file_uploader(
        "Sube tu dataset CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        st.session_state.df_raw = pd.read_csv(uploaded_file)

        df = st.session_state.df_raw

        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", f"{len(df):,}")
        c2.metric("Nulos", f"{df.isnull().sum().sum():,}")
        c3.metric("Duplicados", f"{df.duplicated().sum():,}")

        st.subheader("Diagnóstico")
        st.dataframe(
            pd.DataFrame({
                "Tipo": df.dtypes.astype(str),
                "Nulos": df.isnull().sum(),
                "% Nulos": (df.isnull().sum() / len(df) * 100).round(2)
            }),
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            drop_dups = st.checkbox("Eliminar duplicados", True)
            impute = st.checkbox("Imputar valores faltantes", True)

        with col2:
            clean_age = st.checkbox("Eliminar edades > 100", True)

        if st.button("🛠️ Limpiar dataset"):
            st.session_state.df_clean = clean_data(
                df,
                drop_dups,
                impute,
                clean_age
            )
            st.success(
                f"Dataset procesado: "
                f"{len(st.session_state.df_clean):,} registros."
            )

        if st.session_state.df_clean is not None:
            st.subheader("Dataset limpio")
            st.dataframe(
                st.session_state.df_clean.head(10),
                use_container_width=True
            )

            csv = st.session_state.df_clean.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Descargar CSV limpio",
                csv,
                "credit_risk_cleaned.csv",
                "text/csv"
            )

    else:
        st.info("Sube un CSV para comenzar.")


# ============================================================
# 2. DASHBOARD
# ============================================================
with tab_dashboard:
    st.header("📊 Dashboard de negocio")

    if st.session_state.df_clean is None:
        st.warning("Primero carga y limpia el dataset.")
    else:
        df = st.session_state.df_clean.copy()

        if "loan_status" not in df.columns:
            st.error("El dataset necesita la columna loan_status.")
        else:
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Registros", f"{len(df):,}")
            c2.metric(
                "Tasa de Default",
                f"{df['loan_status'].mean() * 100:.2f}%"
            )

            if "loan_amnt" in df.columns:
                c3.metric(
                    "Préstamo promedio",
                    f"${df['loan_amnt'].mean():,.0f}"
                )

            if "person_income" in df.columns:
                c4.metric(
                    "Ingreso promedio",
                    f"${df['person_income'].mean():,.0f}"
                )

            if "loan_intent" in df.columns:
                st.subheader("Default por propósito del préstamo")

                fig = px.histogram(
                    df,
                    x="loan_intent",
                    color="loan_status",
                    barmode="group"
                )

                st.plotly_chart(fig, use_container_width=True)

            if all(c in df.columns for c in ["person_income", "loan_amnt"]):
                st.subheader("Ingreso vs. monto solicitado")

                sample = df.sample(min(1000, len(df)), random_state=42)

                fig2 = px.scatter(
                    sample,
                    x="person_income",
                    y="loan_amnt",
                    color="loan_status",
                    opacity=0.6
                )

                st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# 3. MODELOS ML + DL
# ============================================================
with tab_models:
    st.header("🤖 Comparación de Machine Learning y Deep Learning")

    if st.session_state.df_clean is None:
        st.warning("Primero carga y limpia el dataset.")
    else:
        df = st.session_state.df_clean.copy()

        target = "loan_status"

        if target not in df.columns:
            st.error("No se encontró loan_status.")
        else:
            # Usar todas las variables disponibles excepto el target.
            # Esto permite aprovechar variables categóricas del dataset.
            X = df.drop(columns=[target])
            y = df[target]

            # Eliminar columnas de texto de alta cardinalidad que no son útiles
            # para este modelo si aparecen.
            drop_cols = [
                c for c in ["person_id", "id", "customer_id"]
                if c in X.columns
            ]
            X = X.drop(columns=drop_cols)

            # Separación train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

            preprocessor = build_preprocessor(X)

            models = {
                "Regresión Logística": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ),

                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                ),

                "XGBoost": xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    eval_metric="logloss"
                ),

                "LightGBM": LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    verbose=-1
                )
            }

            results = []
            trained_models = {}

            with st.spinner("Entrenando modelos..."):

                for name, estimator in models.items():

                    pipe = Pipeline([
                        ("preprocessor", preprocessor),
                        ("model", estimator)
                    ])

                    pipe.fit(X_train, y_train)

                    pred = pipe.predict(X_test)
                    prob = pipe.predict_proba(X_test)[:, 1]

                    results.append(
                        metrics_row(
                            name,
                            y_test,
                            pred,
                            prob
                        )
                    )

                    trained_models[name] = pipe

            results_df = pd.DataFrame(results)

            st.subheader("📊 Resultados ML")

            st.dataframe(
                results_df.style.format({
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1-Score": "{:.3f}",
                    "ROC-AUC": "{:.3f}"
                }),
                use_container_width=True
            )

            fig = px.bar(
                results_df,
                x="Modelo",
                y="ROC-AUC",
                text="ROC-AUC",
                title="Comparación ROC-AUC"
            )

            fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            st.plotly_chart(fig, use_container_width=True)

            best_name = results_df.loc[
                results_df["ROC-AUC"].idxmax(),
                "Modelo"
            ]

            best_auc = results_df["ROC-AUC"].max()

            st.success(
                f"🏆 Mejor modelo ML: **{best_name}** "
                f"con ROC-AUC = **{best_auc:.3f}**"
            )

            st.session_state.models = trained_models
            st.session_state.preprocessor = preprocessor
            st.session_state.model_features = X.columns.tolist()

            # ----------------------------------------------------
            # DEEP LEARNING
            # ----------------------------------------------------
            st.markdown("---")
            st.subheader("🧠 Red Neuronal MLP")

            if not TF_AVAILABLE:
                st.warning(
                    "TensorFlow no está instalado. "
                    "Agrega tensorflow al requirements.txt."
                )
            else:
                X_train_nn = preprocessor.fit_transform(X_train)
                X_test_nn = preprocessor.transform(X_test)

                # Convertir sparse matrix a dense para Keras
                if hasattr(X_train_nn, "toarray"):
                    X_train_nn = X_train_nn.toarray()
                    X_test_nn = X_test_nn.toarray()

                model_nn = Sequential([
                    Dense(
                        64,
                        activation="relu",
                        input_shape=(X_train_nn.shape[1],)
                    ),
                    Dropout(0.30),
                    Dense(32, activation="relu"),
                    Dropout(0.20),
                    Dense(16, activation="relu"),
                    Dense(1, activation="sigmoid")
                ])

                model_nn.compile(
                    optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )

                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True
                )

                with st.spinner("Entrenando red neuronal..."):
                    history = model_nn.fit(
                        X_train_nn,
                        y_train,
                        validation_split=0.20,
                        epochs=50,
                        batch_size=64,
                        callbacks=[early_stop],
                        verbose=0
                    )

                nn_prob = model_nn.predict(
                    X_test_nn,
                    verbose=0
                ).ravel()

                nn_pred = (nn_prob >= 0.5).astype(int)

                nn_auc = roc_auc_score(y_test, nn_prob)

                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric(
                    "Accuracy",
                    f"{accuracy_score(y_test, nn_pred):.3f}"
                )
                c2.metric(
                    "Precision",
                    f"{precision_score(y_test, nn_pred, zero_division=0):.3f}"
                )
                c3.metric(
                    "Recall",
                    f"{recall_score(y_test, nn_pred, zero_division=0):.3f}"
                )
                c4.metric(
                    "F1",
                    f"{f1_score(y_test, nn_pred, zero_division=0):.3f}"
                )
                c5.metric(
                    "ROC-AUC",
                    f"{nn_auc:.3f}"
                )

                history_df = pd.DataFrame(history.history)

                fig_loss = px.line(
                    history_df,
                    y=["loss", "val_loss"],
                    title="Evolución del entrenamiento MLP"
                )

                st.plotly_chart(
                    fig_loss,
                    use_container_width=True
                )

                st.session_state.nn_model = model_nn
                st.session_state.nn_preprocessor = preprocessor


# ============================================================
# 4. SIMULADOR
# ============================================================
with tab_simulator:
    st.header("💰 Simulador de Riesgo y Amortización")

    if not st.session_state.models:
        st.warning(
            "Primero entrena los modelos en la Pestaña 3."
        )
    else:
        model_names = list(st.session_state.models.keys())

        selected_model = st.selectbox(
            "Modelo para evaluar la solicitud",
            model_names,
            index=model_names.index("XGBoost")
            if "XGBoost" in model_names else 0
        )

        model = st.session_state.models[selected_model]

        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input(
                "Edad",
                18,
                90,
                30
            )

            income = st.number_input(
                "Ingreso anual",
                1000,
                1000000,
                45000,
                step=1000
            )

            amount = st.number_input(
                "Monto solicitado",
                500,
                200000,
                12000,
                step=500
            )

        with c2:
            rate = st.number_input(
                "Tasa anual (%)",
                1.0,
                40.0,
                11.5,
                step=0.25
            )

            months = st.slider(
                "Plazo (meses)",
                6,
                84,
                36,
                step=6
            )

        if st.button("📊 Evaluar solicitud"):

            # Crear una fila usando las variables del modelo.
            # Para variables adicionales del dataset usamos valores
            # de referencia (mediana/moda).
            df = st.session_state.df_clean
            row = {}

            for col in st.session_state.model_features:

                if col == "person_age":
                    row[col] = age

                elif col == "person_income":
                    row[col] = income

                elif col == "loan_amnt":
                    row[col] = amount

                elif col == "loan_int_rate":
                    row[col] = rate

                elif pd.api.types.is_numeric_dtype(df[col]):
                    row[col] = df[col].median()

                else:
                    mode = df[col].mode()
                    row[col] = mode.iloc[0] if len(mode) else ""

            input_df = pd.DataFrame([row])

            probability = (
                model.predict_proba(input_df)[0][1] * 100
            )

            monthly_rate = rate / 100 / 12
            monthly_income = income / 12

            if monthly_rate > 0:
                payment = (
                    amount
                    * (
                        monthly_rate
                        * (1 + monthly_rate) ** months
                    )
                    / (
                        (1 + monthly_rate) ** months - 1
                    )
                )
            else:
                payment = amount / months

            dti = payment / monthly_income * 100

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Probabilidad de Default",
                f"{probability:.1f}%"
            )

            c2.metric(
                "Cuota mensual",
                f"${payment:,.2f}"
            )

            c3.metric(
                "DTI",
                f"{dti:.1f}%"
            )

            if probability < 25 and dti <= 35:
                st.success("🟢 APROBACIÓN SUGERIDA")

            elif probability < 50 and dti <= 45:
                st.warning("🟡 REVISIÓN MANUAL")

            else:
                st.error("🔴 RECHAZO SUGERIDO")


# ============================================================
# 5. XAI / SHAP
# ============================================================
with tab_xai:
    st.header("⚡ Explicabilidad del modelo con SHAP")

    if not st.session_state.models:
        st.warning(
            "Primero entrena los modelos en la Pestaña 3."
        )
    else:
        if "XGBoost" not in st.session_state.models:
            st.warning("XGBoost no está disponible.")
        else:
            st.info(
                "SHAP permite analizar qué variables influyen "
                "en las predicciones del modelo."
            )

            xgb_pipeline = st.session_state.models["XGBoost"]
            df = st.session_state.df_clean

            X = df.drop(columns=["loan_status"])

            drop_cols = [
                c for c in ["person_id", "id", "customer_id"]
                if c in X.columns
            ]

            X = X.drop(columns=drop_cols)

            # Transformación de datos
            prep = xgb_pipeline.named_steps["preprocessor"]
            model_xgb = xgb_pipeline.named_steps["model"]

            X_transformed = prep.transform(X)

            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()

            feature_names = prep.get_feature_names_out()

            # Muestra para evitar gráficos demasiado pesados
            sample_size = min(1000, X_transformed.shape[0])

            X_sample = X_transformed[:sample_size]

            explainer = shap.TreeExplainer(model_xgb)

            shap_values = explainer.shap_values(
                X_sample
            )

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            importance = (
                np.abs(shap_values)
                .mean(axis=0)
            )

            shap_df = pd.DataFrame({
                "Variable": feature_names,
                "Importancia SHAP": importance
            }).sort_values(
                "Importancia SHAP",
                ascending=False
            ).head(15)

            st.subheader(
                "📌 Variables más importantes"
            )

            fig_shap = px.bar(
                shap_df.sort_values(
                    "Importancia SHAP"
                ),
                x="Importancia SHAP",
                y="Variable",
                orientation="h",
                title="Importancia global según SHAP"
            )

            st.plotly_chart(
                fig_shap,
                use_container_width=True
            )

            st.caption(
                "Una mayor magnitud SHAP indica mayor influencia "
                "en las predicciones del modelo; la dirección "
                "del efecto se analiza a nivel individual."
            )

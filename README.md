# proyecto_analisis_datos

# 🏦 Credit Risk Analytics & Default Predictor

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

Una aplicación web interactiva desarrollada en **Streamlit** diseñada para analizar la cartera de créditos bancarios de una institución financiera y predecir la probabilidad de incumplimiento de pago (Default) de nuevos solicitantes mediante un modelo de **Machine Learning**.

---

## 🎯 Objetivo del Proyecto

El objetivo principal de este proyecto final es proporcionar una herramienta integral para el área de análisis de riesgo crediticio que permita:
1. **Monitorear KPIs de Negocio:** Evaluar el monto total de la cartera, la tasa general de morosidad y promedios de montos solicitados e ingresos.
2. **Análisis Exploratorio Interactivo:** Filtrar y visualizar la distribución de préstamos por propósito y la relación entre ingresos y deuda.
3. **Evaluación de Riesgo en Tiempo Real:** Simular la aprobación o rechazo de un nuevo crédito mediante un modelo predictivo integrado.

---

## 📊 Características Principales

* **Filtros Dinámicos:** Filtrado en tiempo real por el propósito del préstamo y rango de edad del cliente.
* **Dashboard Visual (Plotly):**
  * Gráfico de distribución de préstamos según su estado (Al día vs. Default).
  * Diagrama de dispersión (*Scatter Plot*) de Ingreso Anual vs. Monto Solicitado.
* **Simulador Predictivo (Machine Learning):** Algoritmo *Random Forest Classifier* entrenado para predecir el porcentaje de riesgo de un solicitante y clasificarlo en:
  * 🟢 **Aprobado** (Riesgo < 25%)
  * 🟡 **Revisión Manual** (Riesgo 25% - 50%)
  * 🔴 **Rechazado** (Riesgo > 50%)

---

## 📂 Estructura del Repositorio

```text
├── credit_risk_dataset.csv  # Dataset sintético/real de Kaggle
├── app.py                   # Código principal del Dashboard en Streamlit
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Documentación del proyecto

# TP2 - Predicción de Fumadores (Smoking) 🚬

Trabajo Práctico 2 - Módulo 2. Proyecto de Machine Learning para predecir si una persona
es fumadora  o no  a partir de datos de un chequeo médico.

# Objetivo

Construir un modelo de clasificación que, a partir de varias características médicas (edad, peso,
presión, colesterol, hemoglobina, etc.), prediga si una persona fuma. El modelo se entrena y
valida con un conjunto etiquetado (50.000 personas) y luego se usa para predecir un conjunto
sin etiquetar.

**Métrica de evaluación:** F1-Score de la clase 1 (fumadores).

##  Estructura del proyecto

```
TP2-smoking/
├── data/
│   ├── raw/          # Datos originales (smoking_train_labeled.xlsx, smoking_unlabeled.xlsx)
│   ├── processed/    # Datos limpios + archivo de entrega (smoking_prediction_entrega.csv)
│   └── external/     # (sin uso)
├── models/           # Modelo entrenado (random_forest.joblib) y umbral (umbral.joblib)
├── notebooks/
│   ├── 01_lectura_y_discovery.ipynb       # Primer contacto con los datos
│   ├── 02_eda.ipynb                        # Análisis exploratorio (gráficos)
│   ├── 03_preprocesamiento.ipynb           # Limpieza, encoding y split
│   ├── 04_entrenamiento_y_optimizacion.ipynb # Entrenamiento y comparación de modelos
│   ├── 05_validacion.ipynb                 # Matriz de confusión e importancia de variables
│   └── 06_prediccion.ipynb                 # Predicción final y exportación
├── app.py              # App interactiva con Streamlit (versión local)
├── vercel-app/         # App deployada en Vercel (web pública)
│   ├── api/predict.py  # Función que sirve la página web y hace la predicción
│   └── pyproject.toml  # Dependencias y entrypoint para Vercel
├── requirements.txt
└── README.md
```

## 🔬 Resumen de los experimentos

**1. Discovery y EDA**
- 50.000 filas, sin datos faltantes ni duplicados.
- Target desbalanceado: ~63% no fuma, ~37% fuma (por eso se evalúa con F1, no con accuracy).
- Se descartaron `ID` (identificador) y `oral` (columna constante, sin información).
- El género es el predictor más fuerte: ~52% de los hombres fuma vs ~6% de las mujeres.

**2. Preprocesamiento**
- Encoding binario de `gender` (F/M → 0/1) y `tartar` (N/Y → 0/1).
- Split: 80% entrenamiento / 20% validación, estratificado por el target.

**3. Comparación de modelos** (F1 de la clase 1 en validación):

| Modelo | F1 |
|---|---|
| Árbol de Decisión | 0,653 |
| KNN | 0,634 |
| Gradient Boosting (optimizado con GridSearchCV) | 0,697 |
| **Random Forest** | **0,737** |

- Un solo Árbol de Decisión sufría overfitting (F1 entrenamiento = 1,0 vs validación = 0,653).
- El **Random Forest** (bagging, 200 árboles) fue el mejor modelo.

**4. Optimización del umbral**
- En lugar del umbral por defecto (0,5), se buscó el óptimo para F1: **0,39**.
- Esto subió el F1 de **0,737 a 0,7546**.

## ✅ Resultados y conclusiones

- **Modelo final:** Random Forest + umbral de decisión 0,39.
- **F1-Score en validación:** ~0,75 (recall 0,91, precision 0,65).
- El modelo prioriza **detectar fumadores** (recall alto) a costa de algunas falsas alarmas.
- Las variables más importantes (`gender`, `Gtp`, `hemoglobin`, `height`, `triglyceride`)
  coinciden con lo observado en el análisis exploratorio, lo que da confianza en el modelo.
- La predicción final sobre los 5.692 datos sin etiquetar se encuentra en
  `data/processed/smoking_prediction_entrega.csv`, en la columna `smoking_prediction`.

## 🌐 App web interactiva (bonus)

Además de las notebooks, el proyecto incluye una **aplicación web** donde se pueden
cargar los datos de una persona y obtener la predicción del modelo en vivo.

🔗 **App online:** https://tp-2-smoking.vercel.app/

### ¿Cómo funciona?

La app está formada por:
- Una **página web** (HTML) con deslizadores para los datos de la persona y un botón de predecir.
- Una **función Python** (`vercel-app/api/predict.py`) que sirve esa página y realiza la predicción.
- Una **versión liviana** del Random Forest, **embebida** dentro de la función (no se reentrena nada).

**Flujo paso a paso:**
1. La persona entra a la URL → la función devuelve la página web (deslizadores + botón).
2. La persona ajusta los valores (género, edad, altura, peso, hemoglobina, triglicéridos, Gtp) y
   presiona **¿Es fumador?**.
3. El navegador envía esos datos (en formato JSON) a la función Python.
4. La función:
   - Parte de los valores **medianos** de las 24 variables (las no ingresadas se completan con valores típicos).
   - Reemplaza con los datos que cargó la persona y arma la fila en el orden correcto.
   - Calcula la **probabilidad** de fumar con `predict_proba` y la compara con el **umbral 0,39**.
5. Devuelve el resultado (probabilidad + fuma/no fuma), que se muestra en pantalla.

### Tecnologías

- **Vercel** (hosting gratuito, conectado a GitHub: cada `push` actualiza la app).
- **Streamlit** (`app.py`): una versión alternativa de la app para correr localmente con
  `streamlit run app.py`.

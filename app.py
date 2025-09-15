import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import io
import graphviz

# --- Funciones para el Diccionario de Palabras Spam ---
def cargar_diccionario_manual():
    if os.path.exists("diccionario_manual.pkl"):
        return joblib.load("diccionario_manual.pkl")
    else:
        return []

def guardar_diccionario_manual(diccionario):
    joblib.dump(diccionario, "diccionario_manual.pkl")

# --- 1. SOLUCIÓN DE LA ACTIVIDAD: Modelo de Árbol de Decisión ---
st.title("🛡️ Detector de Spam: Solución de la Actividad")


# Función para entrenar el modelo
def entrenar_modelo_actividad():
    if not os.path.exists('emails.csv'):
        st.error("Error: No se encontró el archivo 'emails.csv'. Por favor, asegúrate de que esté en la misma carpeta.")
        st.stop()
    
    df = pd.read_csv('emails.csv')
    X = df.drop(columns=['Email No.', 'Prediction'], errors='ignore')
    y = df['Prediction']
    
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    modelo_arbol = DecisionTreeClassifier(
        criterion='entropy', max_depth=5, random_state=42
    )
    modelo_arbol.fit(X_entrenamiento, y_entrenamiento)
    
    joblib.dump(modelo_arbol, 'modelo_actividad.pkl')
    joblib.dump(X.columns.tolist(), 'diccionario_actividad.pkl')
    
    return modelo_arbol, X_prueba, y_prueba, X.columns.tolist()

# Cargar o entrenar el modelo de la actividad
try:
    modelo_actividad = joblib.load('modelo_actividad.pkl')
    diccionario_actividad = joblib.load('diccionario_actividad.pkl')
    
    df_temp = pd.read_csv('emails.csv')
    X_temp = df_temp.drop(columns=['Email No.', 'Prediction'], errors='ignore')
    y_temp = df_temp['Prediction']
    _, X_prueba, _, y_prueba = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

except FileNotFoundError:
    modelo_actividad, X_prueba, y_prueba, diccionario_actividad = entrenar_modelo_actividad()
    st.success("¡Modelo de la actividad entrenado y archivos creados exitosamente!")

# Evaluación del Modelo
st.header("1.1 Evaluación del Modelo")
y_pred = modelo_actividad.predict(X_prueba)
accuracy = accuracy_score(y_prueba, y_pred)
precision = precision_score(y_prueba, y_pred)
recall = recall_score(y_prueba, y_pred)
f1 = f1_score(y_prueba, y_pred)
conf_matrix = confusion_matrix(y_prueba, y_pred)

st.write("### Métricas de Rendimiento")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Exactitud", f"{accuracy:.2f}")
with col2:
    st.metric("Precisión", f"{precision:.2f}")
with col3:
    st.metric("Recall", f"{recall:.2f}")
with col4:
    st.metric("F1-Score", f"{f1:.2f}")
    
st.write("### Matriz de Confusión")
st.dataframe(pd.DataFrame(conf_matrix, index=['No Spam Real', 'Spam Real'], columns=['No Spam Predicción', 'Spam Predicción']))

# Visualización del Árbol de Decisión
st.header("1.2 Visualización del Árbol de Decisión")
st.info("Este árbol muestra cómo el modelo toma decisiones. Los nodos se dividen en base a palabras clave de la base de datos.")
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    modelo_actividad,
    feature_names=diccionario_actividad,
    class_names=['No Spam', 'Spam'],
    filled=True,
    ax=ax,
    max_depth=3,
    fontsize=8
)
st.pyplot(fig)


# --- 2. MI CLASIFICADOR: Basado en Diccionario ---
st.title("💡 Mi Clasificador de Spam Personalizado")


st.header("2.1 Crear y Guardar un Diccionario de Palabras Spam")
st.info("Aquí puedes añadir y guardar tus propias palabras clave para usar en la clasificación.")

palabras_spam_texto = st.text_area(
    "Ingresa tus palabras clave aquí (separadas por comas):",
    height=150,
    value="oferta, gratis, promoción, urgente, gana, dinero, clic, ahora, regalo, exclusivo, especial, oportunidad"
)

if st.button("Guardar Diccionario"):
    diccionario_nuevo = [palabra.strip().lower() for palabra in palabras_spam_texto.split(',')]
    diccionario_nuevo = sorted(list(set([p for p in diccionario_nuevo if p])))
    guardar_diccionario_manual(diccionario_nuevo)
    st.success(f"Diccionario guardado exitosamente. Se guardaron {len(diccionario_nuevo)} palabras.")

diccionario_manual = cargar_diccionario_manual()

st.header("2.2 Analiza un Correo Electrónico")
st.info("Escribe un correo para ver si contiene alguna de las palabras de tu diccionario.")

if not diccionario_manual:
    st.warning("⚠️ Primero debes crear y guardar un diccionario de palabras en la sección 2.")
else:
    texto_correo = st.text_area("Ingresa el texto del correo aquí:", height=200, key='texto_correo_manual')

    if st.button("Clasificar Correo"):
        if texto_correo:
            palabras_en_correo = set(texto_correo.lower().split())
            es_spam = False
            palabras_encontradas = []
            for palabra in palabras_en_correo:
                if palabra in diccionario_manual:
                    es_spam = True
                    palabras_encontradas.append(palabra)
            
            st.write("---")
            st.subheader("Resultado de la Clasificación:")
            if es_spam:
                st.error("¡Atención! Este correo es **SPAM**")
                st.write(f"Palabras clave de spam encontradas: **{', '.join(palabras_encontradas)}**")
            else:
                st.success("¡Tranquilo! Este correo es **NO SPAM**")
                st.write("No se encontraron palabras clave de spam en el correo.")
        else:
            st.warning("Por favor, ingresa el texto del correo para clasificar.")
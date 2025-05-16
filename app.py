from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class SimpleFCM:
    def __init__(self, adjacency_matrix, threshold_function=None):
        self.W = np.array(adjacency_matrix)
        self.n = self.W.shape[0]
        self.threshold = threshold_function if threshold_function else self.sigmoid

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def infer(self, initial_state, steps=10):
        state = np.array(initial_state, dtype=float)
        history = [state.copy()]
        for _ in range(steps):
            state = self.threshold(np.dot(state, self.W))
            history.append(state.copy())
        return history
    
# Cargar matriz del modelo FCM
with open("mi_proyecto_fetal/modelos/fcm_model.pkl", "rb") as f:
    fcm_data = pickle.load(f)

# Inicializar el modelo FCM
modelo_fcm = SimpleFCM(adjacency_matrix=fcm_data["adjacency_matrix"])

app = Flask(__name__,
            template_folder="mi_proyecto_fetal/templates",
            static_folder="mi_proyecto_fetal/static")

# Carga de modelos
scaler = pickle.load(open("mi_proyecto_fetal/modelos/escalador.pkl", "rb"))
modelos = {
    "logistica": pickle.load(open("mi_proyecto_fetal/modelos/Regresion_logistica.pkl", "rb")),
    "mlp": pickle.load(open("mi_proyecto_fetal/modelos/red_neuronal.pkl", "rb")),
    "svm": pickle.load(open("mi_proyecto_fetal/modelos/svm.pkl", "rb")),
}

def predecir_con_fcm(entrada):
    estados = modelo_fcm.infer(initial_state=np.array(entrada).flatten(), steps=10)
    final = estados[-1]
    return 1 if np.mean(final) > 0.5 else 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form_individual')
def form_individual():
    return render_template('form_individual.html')


@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos = [float(request.form[f'dato{i}']) for i in range(1, 31)]
        modelo_nombre = request.form['modelo']

        if modelo_nombre not in modelos and modelo_nombre != "fcm":
            return render_template('form_individual.html', resultado="Modelo no válido seleccionado.")

        datos_array = np.array([datos])
        datos_array = scaler.transform(datos_array)  # Escalado obligatorio

        if modelo_nombre == "fcm":
            prediccion = predecir_con_fcm(datos_array)
        else:
            modelo = modelos[modelo_nombre]
            prediccion = modelo.predict(datos_array)[0]

        return render_template(
            'form_individual.html',
            resultado={"prediccion": prediccion, "modelo": modelo_nombre}
        )

    except Exception as e:
        return render_template('form_individual.html', resultado=f"Error en la predicción: {e}")
    
    
@app.route('/form_lote')
def form_lote():
    return render_template('form_lote.html')

@app.route('/evaluar', methods=['GET', 'POST'])
def evaluar_modelo():
    if request.method == 'POST':
        archivo = request.files['archivo']
        modelo_nombre = request.form['modelo']

        if not archivo:
            return render_template('form_lote.html', resultados="No se subió ningún archivo.")

        try:
            # Leer dataset (soporte para Excel y CSV)
            if archivo.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(archivo)
            else:
                df = pd.read_csv(archivo)
                
            # Verificar columnas necesarias
            if "C31" not in df.columns:
                return render_template('form_lote.html', error="El archivo no contiene la columna 'C31' requerida.")

            X = df.drop(columns=["C31"])
            y = df["C31"]

            # Escalar
            X_scaled = scaler.transform(X)

            # Predecir
            if modelo_nombre == "fcm":
                y_pred = [predecir_con_fcm([fila]) for fila in X_scaled]
            else:
                modelo = modelos[modelo_nombre]
                y_pred = modelo.predict(X_scaled)

            # Métricas
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            classification_report_str = classification_report(y, y_pred)
            cm = confusion_matrix(y, y_pred)

            # Matriz de confusión
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Matriz de Confusión")
            plt.xlabel("Predicción")
            plt.ylabel("Real")
            plt.tight_layout()
            os.makedirs("static", exist_ok=True)
            plt.savefig("static/confusion_matrix.png")
            plt.close()

            return render_template("form_lote.html", 
                                resultados=True,
                                modelo_nombre=modelo_nombre.upper(),
                                archivo_nombre=archivo.filename,
                                accuracy=accuracy,
                                precision=precision,
                                recall=recall,
                                f1=f1,
                                classification_report=classification_report_str,
                                cm=cm)

        except Exception as e:
            return render_template('form_lote.html', error=f"Error al procesar el archivo: {str(e)}")

    return render_template("form_lote.html")


if __name__ == '__main__':
    app.run(debug=True)
    



import numpy as np
import pickle
import streamlit as st

# Path del modelo preentrenado
MODEL_PATH = 'ingreso_model.pkl'

# Se recibe los datos de entrada y el modelo, devuelve la predicción
def model_prediction(x_in, model):
    x = np.asarray(x_in).reshape(1, -1)
    preds = model.predict(x)
    return preds

def main():
    model = None

    # Se carga el modelo
    if model is None:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)

    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">PREDICCIÓN DE SALARIO</h1>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Lectura de datos
    Edad = st.text_input("Edad:")
    ClaseTrabajo = st.text_input("Clase de Trabajo:")
    PesoFinal = st.text_input("Peso Final:")
    Educacion = st.text_input("Educación:")
    NumEducacion = st.text_input("Número de Educación:")
    EstadoCivil = st.text_input("Estado Civil:")
    Ocupacion = st.text_input("Ocupación:")
    Relacion = st.text_input("Relación:")
    Raza = st.text_input("Raza:")
    GananciaCapital = st.text_input("Ganancia de Capital:")
    PerdidaCapital = st.text_input("Pérdida de Capital:")
    HorasPorSemana = st.text_input("Horas por Semana:")
    Pais = st.text_input("País:")
    Sexo = st.text_input("Sexo (True para Hombre, False para Mujer):")

    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción"):
        try:
            x_in = [
                int(Edad), int(ClaseTrabajo), int(PesoFinal), int(Educacion), int(NumEducacion),
                int(EstadoCivil), int(Ocupacion), int(Relacion), int(Raza), 
                int(GananciaCapital), int(PerdidaCapital), int(HorasPorSemana), 
                int(Pais), bool(Sexo)
            ]
            predicts = model_prediction(x_in, model)
            if predicts[0]:
                st.success('El salario predicho es: >50K')
            else:
                st.success('El salario predicho es: <=50K')
        except ValueError:
            st.error("Por favor, introduce todos los valores correctamente.")

if __name__ == '__main__':
    main()

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
    Edad = st.number_input("Edad:", min_value=0, max_value=100, step=1)
    ClaseTrabajo = st.selectbox("Clase de Trabajo:", [
        "Federal-gov (1)", "Local-gov (2)", "Never-worked (3)", "Private (4)", 
        "Self-emp-inc (5)", "Self-emp-not-inc (6)", "State-gov (7)", "Without-pay (8)"
    ])
    PesoFinal = st.number_input("Peso Final:", min_value=0, step=1)
    Educacion = st.selectbox("Educación:", [
        "10th (1)", "11th (2)", "12th (3)", "1st-4th (4)", "5th-6th (5)", "7th-8th (6)", 
        "9th (7)", "Assoc-acdm (8)", "Assoc-voc (9)", "Bachelors (10)", "Doctorate (11)", 
        "HS-grad (12)", "Masters (13)", "Preschool (14)", "Prof-school (15)", "Some-college (16)"
    ])
    NumEducacion = st.number_input("Número de Educación:", min_value=0, step=1)
    EstadoCivil = st.selectbox("Estado Civil:", [
        "Divorced (1)", "Married-AF-spouse (2)", "Married-civ-spouse (3)", 
        "Married-spouse-absent (4)", "Never-married (5)", "Separated (6)", "Widowed (7)"
    ])
    Ocupacion = st.selectbox("Ocupación:", [
        "Adm-clerical (1)", "Armed-Forces (2)", "Craft-repair (3)", "Exec-managerial (4)", 
        "Farming-fishing (5)", "Handlers-cleaners (6)", "Machine-op-inspct (7)", "Other-service (8)", 
        "Priv-house-serv (9)", "Prof-specialty (10)", "Protective-serv (11)", "Sales (12)", 
        "Tech-support (13)", "Transport-moving (14)"
    ])
    Relacion = st.selectbox("Relación:", [
        "Husband (1)", "Not-in-family (2)", "Other-relative (3)", "Own-child (4)", 
        "Unmarried (5)", "Wife (6)"
    ])
    Raza = st.selectbox("Raza:", [
        "Amer-Indian-Eskimo (1)", "Asian-Pac-Islander (2)", "Black (3)", 
        "Other (4)", "White (5)"
    ])
    GananciaCapital = st.number_input("Ganancia de Capital:", min_value=0, step=1)
    PerdidaCapital = st.number_input("Pérdida de Capital:", min_value=0, step=1)
    HorasPorSemana = st.number_input("Horas por Semana:", min_value=0, max_value=168, step=1)
    Pais = st.selectbox("País:", [
        "Cambodia (1)", "Canada (2)", "China (3)", "Columbia (4)", "Cuba (5)", 
        "Dominican-Republic (6)", "Ecuador (7)", "El-Salvador (8)", "England (9)", 
        "France (10)", "Germany (11)", "Greece (12)", "Guatemala (13)", "Haiti (14)", 
        "Hong (15)", "Holand-Netherlands (16)", "Honduras (17)", "Hungary (18)", 
        "India (19)", "Ireland (20)", "Iran (21)", "Italy (22)", "Jamaica (23)", 
        "Japan (24)", "Laos (25)", "Mexico (26)", "Nicaragua (27)", "North (28)", 
        "Outlying-US(Guam-USVI-etc) (29)", "Peru (30)", "Philippines (31)", "Poland (32)", 
        "Portugal (33)", "Puerto-Rico (34)", "Scotland (35)", "South (36)", "Taiwan (37)", 
        "Thailand (38)", "United-States (39)", "Trinadad&Tobago (40)", "Vietnam (41)", 
        "Yugoslavia (42)"
    ])
    Sexo = st.selectbox("Sexo:", ["Hombre (True)", "Mujer (False)"])

    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción"):
        try:
            x_in = [
                Edad, int(ClaseTrabajo.split(' ')[-1][1]), PesoFinal, int(Educacion.split(' ')[-1][1]), 
                NumEducacion, int(EstadoCivil.split(' ')[-1][1]), int(Ocupacion.split(' ')[-1][1]), 
                int(Relacion.split(' ')[-1][1]), int(Raza.split(' ')[-1][1]), 
                GananciaCapital, PerdidaCapital, HorasPorSemana, 
                int(Pais.split(' ')[-1][1]), Sexo == "Hombre (True)"
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

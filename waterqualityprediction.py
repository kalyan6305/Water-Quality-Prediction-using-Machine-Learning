import streamlit as st
import pickle
import pandas as pd


# Load the models
dt_model = pickle.load(open('dt_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))

def main():
    st.title("Water Quality Prediction Application")
    st.write("This is a Machine Learning application to predict water potability.")
    st.subheader("Prediction")

    # Input fields for water quality parameters
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
    solids = st.number_input("Solids", min_value=0.0, value=20000.0)
    chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
    sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
    conductivity = st.number_input("Conductivity", min_value=0.0, value=500.0)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=70.0)
    turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

    if st.button("Predict"):
        # Create a dataframe with the input data
        input_data = [[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]]
        feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        input_df = pd.DataFrame(input_data, columns=feature_names)

        # Prediction using Decision Tree
        dt_prediction = dt_model.predict(input_df)
        if dt_prediction[0] == 0:
            st.success("Decision Tree Prediction: The water is not potable.")
        else:
            st.success("Decision Tree Prediction: The water is potable.")

        # Prediction using KNN
        knn_prediction = knn_model.predict(input_df)
        if knn_prediction[0] == 0:
            st.success("KNN Prediction: The water is not potable.")
        else:
            st.success("KNN Prediction: The water is potable.")

main()
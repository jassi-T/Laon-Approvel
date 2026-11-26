import streamlit as st
import pandas as pd

st.title("Machine Learning Project")

st.subheader("ML Prediction Dashboard")

st.write("Welcome to my Machine Learning project!")

age = st.number_input("Enter Age", min_value=1, max_value=100)

salary = st.number_input("Enter Salary", min_value=0)

if st.button("Predict"):
    st.write("Age:", age)
    st.write("Salary:", salary)

import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import tensorflow as tf

purchase_model = tf.keras.models.load_model("ann_model.h5")
st.title("##Purchased Model")
Age=st.text_input("Enter your age:")
Estimated_salary=st.text_input("Enter your Salary:")
if st.button("prediction"):
    data=[[Age,Estimated_salary]]
    data_array=np.array(data,dtype=float).reshape(1,-1)
    prediction=purchase_model.predict(data_array)
    if prediction == 1:
        st.write("##Purchased")
    else:
        st.write("##Not purchased")


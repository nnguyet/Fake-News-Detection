#from os import name
import streamlit as st
# import pandas as pd
# import pickle as pkl
import joblib
import __main__ 


if __name__ == "__main__":
    st.title ("FAKE NEWS DETECTION")

    left, right = st.columns(2)
    form = left.form("input news")
    news = form.text_input("Input news")
    model = form.selectbox("Choose model", ["MLP Classifier","DecisionTree"], index = 0)
    predict = form.form_submit_button("Predict news")
    pickle_in = open('test.pkl', 'rb') 
    classifier = joblib.load(pickle_in)
    pickle_in.close()
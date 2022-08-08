import streamlit as st
import pandas as pd 
import numpy as np


from PIL import Image

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:

	pic = Image.open(uploaded_file)
	pic1 = pic.convert("L")
	pic1 = pic1.resize((28,28))
	pic1 = np.array(pic1)

	st.image(uploaded_file)

btn_click = st.button("PREDICT")

if btn_click == True:

	from pickle import load

	log_classifier = load(open('C:/Users/YASH/Desktop/mnist_data-29-4-2022/productionization/classifier.pkl','rb'))

	pred = log_classifier.predict(pic1.reshape(1,-1))

	st.text("The Number is : ")

	st.title(pred[0])


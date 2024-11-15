#This is the main streamlit file
import streamlit as st

# Page header
container = st.container()
col1, col2, col3 = container.columns([1, 1, 1])  # Adjust proportions of columns
col2.image(r"assets/BeCodeHarmonyRealEstateLogo.jpg", width=80)

# Page header
container = st.container()
col1, col2, col3 = container.columns([1, 3, 1])  # Adjust proportions of columns

col2.subheader("Harmony Real Estate")
col2.write("House price analysis and prediction Suite")

st.image(r"assets/house.jpg", width = 600)

import streamlit as st

# Page header
container = st.container()
col1, col2 = container.columns([1, 5])  # Adjust proportions of columns
col1.image(r"assets/BeCodeHarmonyRealEstateLogo.jpg", width=60)
col2.subheader("Dataset Cleaning", divider="green")
#This is the main streamlit file
import streamlit as st
#from streamlit_multipage import MultiPage
import pages.collection as collection
import pages.cleaning as cleaning
import pages.exploratory_analysis as exploratory_analysis
import pages.model_building as model_building
import pages.quick_predict as quick_predict
import pages.select_predict as select_predict

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["Data collection", "Cleaning", "Exploratory analysis", "Model building", "Prediction"])

# Show the selected page
if page == "Data collection":
    collection.app()
elif page == "Cleaning":
    cleaning.app()
elif page == "Exploratory analysis":
    exploratory_analysis.app()
elif page == "Model building":
    model_building.app()
elif page == "Prediction":
    select_predict.app()

"""# Create a MultiPage instance
app = MultiPage()

# Register each page with a title and the module

app.add_app("Data collection", 1_collection.app)
app.add_app("Cleaning", 2_cleaning.app)
app.add_app("Exploratory analysis", 3_exploratory_analysis.app)
app.add_app("Model building", 4_model_building.app)
app.add_app("Select predict", 5_quick_predict.app)
app.add_app("Quick predict", 6_select_predict.app)

# Run the app
app.run()

st.sidebar.image(r"BeCodeHarmonyRealEstateLogo.jpg", width=100)
st.write("Start page")

"""
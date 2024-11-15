from streamlit_multipage import MultiPage
import pages.collection as collection
import pages.data_cleaning as cleaning
import pages.exploratory_analysis as exploratory_analysis
import pages.model_building as model_building
import drafts.quick_predict as quick_predict
import pages.prediction as prediction


"""# Sidebar
st.sidebar.title("Immo Eliza")


st.sidebar.subheader('Web Scraper')
st.sidebar.subheader('Data Cleaner')
st.sidebar.subheader('Data Analyzer')
st.sidebar.subheader('Modeller')
st.sidebar.subheader('Predictor')"""

# Page header
container1 = st.container()
col1,col2, col3=container1.columns([1,1,1])
#col2.image(r"assets\house.jpg", width = 200)
col2.subheader("Prediction tool with multiple models")


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
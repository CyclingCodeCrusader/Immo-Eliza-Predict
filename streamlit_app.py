#THis is the main streamlit file
import streamlit as st
from streamlit_multipage import MultiPage
import pages.home as home
import pages.about as about
import pages.contact as contact

# Create a MultiPage instance
app = MultiPage()

# Register each page with a title and the module
app.add_app("Home", home.app)
app.add_app("About", about.app)
app.add_app("Contact", contact.app)

# Run the app
app.run()

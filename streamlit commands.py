import streamlit as st
import time


st.title("Immo Eliza")
# Sidebars and containers
st.sidebar.title("Prediction tool")
st.sidebar.markdown(""This prediction tool is powered by Harmony Real Estate."")

#st.caption("This is the caption")
#st.code("x = 2021")
#st.latex(r''' a+a r^1+a r^2+a r^3 ''')


st.sidebar.write("Hello, and welcome to the prediction tool of Immo ELiza. Below you can enter the information of the property you want to buy oir sell. The prediction tool will return an estimate of what the ask price on ImmoWeb can be.")

st.subheader("House information")
st.image("assets\house.jpg", caption="A house for sale")
#st.audio("audio.mp3")
#st.video("video.mp4")

# Widgets
st.subheader('Widgets')
st.checkbox('Yes')
st.button('Click Me')
st.radio('Pick your gender', ['Male', 'Female'])
st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange'])
st.multiselect('Choose a planet', ['Jupiter', 'Mars', 'Neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0, 50)

# Input
st.subheader('Input')
st.number_input('Pick a number', 0, 10)
st.text_input('Email address')
st.date_input('Traveling date')
st.time_input('School time')
st.text_area('Description')
st.file_uploader('Upload a photo')
st.color_picker('Choose your favorite color')

# Progress
st.subheader('Progress')
st.balloons()  # Celebration balloons
st.progress(10)  # Progress bar
with st.spinner('Wait for it...'):
    time.sleep(10)  # Simulating a process delay

# Status
st.subheader('Status returns')
st.success("You did it!")
st.error("Error occurred")
st.warning("This is a warning")
st.info("It's easy to build a Streamlit app")
st.exception(RuntimeError("RuntimeError exception"))

# Sidebars and containers
st.sidebar.title("Sidebar Title")
st.sidebar.markdown("This is the sidebar content")

with st.container():
    st.write("This is inside the container")

container = st.container()
container.write("This is inside the container")
st.write('This is outside the container')

# Visualization
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Histogram plot
rand = np.random.normal(1, 2, size=20)
fig, ax = plt.subplots()
ax.hist(rand, bins=15)
st.pyplot(fig)

# Line graph
df = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.line_chart(df)

# Bar chart
df = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.bar_chart(df)

# Area chart
df = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.area_chart(df)

# Altiar chart
import altair as alt

df = pd.DataFrame(np.random.randn(500, 3), columns=['x', 'y', 'z'])
chart = alt.Chart(df).mark_circle().encode(
    x='x', y='y', size='z', color='z', tooltip=['x', 'y', 'z']
)
st.altair_chart(chart, use_container_width=True)

# Graphics
import graphviz

st.graphviz_chart('''
    digraph {
        Web Scraper -> Data Cleaner
        Data Cleaner -> Data Analyzer
        Data Analyzer -> Modeller
        Modeller -> Predictor
    }
''')

# Display maps
import pandas as pd
import numpy as np
import streamlit as st

df = pd.DataFrame(
    np.random.randn(500, 2) / [50, 50] + [37.76, -122.4], columns=['lat', 'lon']
)
st.map(df)
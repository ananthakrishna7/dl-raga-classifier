import streamlit as st
from main import *

homePage = st.Page("./pages/home.py", title="Home", url_path="/")
modelPage = st.Page("./pages/model_metrics.py", title="Model Metrics", url_path="/model")
uploadPage = st.Page("./pages/upload.py", title="Uploads", url_path="/upload")

pg = st.navigation([homePage, modelPage, uploadPage])

pg.run()

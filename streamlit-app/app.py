import streamlit as st

# Set page config with font settings
st.set_page_config(
    page_title="Home Page",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom HTML to ensure font is applied
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400..800;1,400..800&display=swap');
    
    /* Force EB Garamond on all text elements */
    * {
        font-family: 'EB Garamond', serif !important;
    }
    
    .stApp h1, .stApp h2, .stApp h3, .stApp p {
        font-family: 'EB Garamond', serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your content
st.title("Home Page")
st.write("Welcome to the Home Page!")

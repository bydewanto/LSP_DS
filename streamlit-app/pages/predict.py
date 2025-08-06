import streamlit as st

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

st.title("Predict Page")
st.write("This is the Predict Page.")
import streamlit as st
import pandas as pd
import os
import sys

# ----------------------------------------------------------------------
# 1. CUSTOMIZATION REQUIRED: STUDENT EDITS THESE TWO LINES ONLY
# ----------------------------------------------------------------------

# ⚠️ CHANGE 1: Set the key name for YOUR data (e.g., 'st05_df', 'st12_df', etc.)
# IMPORTANT: This key MUST match the 'stN_df' key used in the main Home Page
# and accessed by your analysis file.
STUDENT_DATA_KEY = 'st07_df' 

# ⚠️ CHANGE 2: Set the file path to YOUR Streamlit page (e.g., 'pages/07_⚽_Sports_Analysis.py')
STUDENT_PAGE_PATH = 'pages/07_⚽_Sports_Analysis.py' 

# ----------------------------------------------------------------------
# 2. DATA LOADING & REDIRECTION LOGIC (DO NOT EDIT BELOW THIS LINE)
# ----------------------------------------------------------------------

@st.cache_data
def load_student_data(key_name):
    """Loads only the single required CSV file for testing."""
    
    # NEW LOGIC: Convert the session state key (e.g., 'st7_df') to the filename (e.g., 'st07_data.csv')
    # This ensures the student page finds the correct key, but the loader finds the correct file.
    file_root = key_name.replace('_df', '_data') 
    data_path = os.path.join('data', f"{file_root}.csv")
    
    if os.path.exists(data_path):
        st.success(f"Successfully loaded data from {data_path} for testing.")
        return pd.read_csv(data_path)
    else:
        st.error(f"Data file not found at {data_path}. Please check your key or run 'generate_data.py'.")
        return pd.DataFrame()

# Initialize session state with the single data required by this student's page
if 'student_data' not in st.session_state:
    st.session_state['student_data'] = {
        STUDENT_DATA_KEY: load_student_data(STUDENT_DATA_KEY)
    }

# Ensure the main Streamlit configuration is set
st.set_page_config(
    page_title="LOCAL TEST RUNNER", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Local Test Mode")
st.sidebar.warning("This is a temporary file for local testing only.")

# --- Manual Redirection ---
# In a real MPA, this isn't needed, but for testing, we tell the user what to run.
if 'streamlit' in sys.modules:
    st.title("Local Test Runner Setup Complete")
    st.info(f"""
    Setup complete! The required data (`{STUDENT_DATA_KEY}`) is now in session state.
    
    To view your actual project, please select it on the sidebar.""")
    st.stop()
import streamlit as st
import pandas as pd

# The actual page content is executed here by Streamlit
st.title("⚽ Student A: Sports Results Analysis")
st.markdown("---")

# Retrieve shared data from the Home page's session state
if 'student_data' not in st.session_state or st.session_state['student_data']['st07_df'].empty:
    st.warning("Data not loaded. Please ensure the main Home Page ran successfully and the data files exist.")
else:
    df = st.session_state['student_data']['st07_df']

    # --- Student Introductory Section ---
    st.header("1. Introduction and Project Goal")
    st.markdown("""
        **Data Description:** This dataset contains **sports game results** for five fictional teams over two seasons, including scores, attendance figures, and the final result (Win, Loss, Draw).
        
        **Question:** How does **home team performance** vary across different teams, and what is the relationship between the game result and **audience attendance**?
        
        **Interaction:** Use the selection box below to choose a specific team and view its key performance indicators (KPIs) and attendance metrics.
    """)
    st.markdown("---")
    
    # --- Analysis Controls (Moved from Sidebar to Main Page) ---
    col_control, col_spacer = st.columns([1, 3])
    with col_control:
        team_filter = st.selectbox(
            "Select Team to Analyze (Home Games Only):", 
            df['Home_Team'].unique()
        )
    
    # Filter data for the selected team (as Home Team)
    team_df = df[df['Home_Team'] == team_filter]
    
    # --- Analysis Content ---
    if team_df.empty:
        st.info(f"No home games found for the team '{team_filter}' in the dataset to analyze.")
    else:
        st.subheader(f"2. Performance Metrics for the {team_filter}")
        
        col1, col2 = st.columns(2)
        
        # Win/Loss Count
        result_counts = team_df['Result'].value_counts()
        
        with col1:
            st.metric(
                label="Total Home Games Analyzed", 
                value=len(team_df)
            )
            st.dataframe(result_counts.to_frame(), use_container_width=True)
            
        # Average Attendance
        with col2:
            avg_attendance = team_df['Attendance'].mean()
            st.metric(
                label="Average Home Game Attendance", 
                value=f"{avg_attendance:,.0f}"
            )
            st.bar_chart(team_df.set_index('Game_ID')['Attendance'])
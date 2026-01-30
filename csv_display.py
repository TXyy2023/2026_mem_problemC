import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

# Page Config
st.set_page_config(layout="wide", page_title="MCM Results Viewer")

# File Paths
RESULTS_FILE = '/Users/a1234/Desktop/美赛/MCM_Problem_C_Results.csv'
# Original data might be useful, but let's stick to Results for now as requested for "calculated data"
# We can load original for extra context if needed, but the user focused on the output tables.
# User said "Show two tables", implies Result and Original?
# "Show data from both tables".
ORIGINAL_FILE = '/Users/a1234/Desktop/美赛/2026_MCM_Problem_C_Data.csv'

@st.cache_data
def load_data():
    results_df = pd.read_csv(RESULTS_FILE)
    original_df = pd.read_csv(ORIGINAL_FILE)
    return results_df, original_df

def parse_range(range_str):
    """Parses 'Min-Max' or 'Min%-Max%' into tuple (min, max)."""
    if pd.isna(range_str):
        return None, None
    
    # Check for percentage
    is_percent = '%' in str(range_str)
    
    # Remove %
    clean_str = str(range_str).replace('%', '')
    parts = clean_str.split('-')
    
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except:
            return None, None
    return None, None

def main():
    st.title("MCM Problem C - Analysis Dashboard")
    
    try:
        results_df, original_df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar
    st.sidebar.header("Filter Options")
    
    # Season Selector
    seasons = sorted(results_df['Season'].unique())
    selected_season = st.sidebar.selectbox("Select Season", seasons)
    
    # Filter Data
    season_results = results_df[results_df['Season'] == selected_season]
    season_original = original_df[original_df['season'] == selected_season]
    
    # View Mode
    view_mode = st.sidebar.radio("View Mode", ["Season Overview", "Celebrity Detail"])
    
    if view_mode == "Season Overview":
        st.header(f"Season {selected_season} Overview")
        
        # Display Tabs
        tab1, tab2 = st.tabs(["Calculated Results", "Original Data"])
        
        with tab1:
            st.subheader("Calculated Rankings & Ranges")
            st.dataframe(season_results, use_container_width=True)
            
            # Download button
            # st.download_button(...)
            
        with tab2:
            st.subheader("Original Raw Data")
            st.dataframe(season_original, use_container_width=True)
            
        # Charts
        st.subheader("Performance Trends")
        
        # Average Score Chart
        avg_scores = season_results.groupby('CelebrityName')['JudgeScore'].mean().sort_values()
        fig_avg = px.bar(avg_scores, orientation='h', title="Average Judge Score by Celebrity")
        st.plotly_chart(fig_avg, use_container_width=True)
        
    elif view_mode == "Celebrity Detail":
        celebrities = sorted(season_results['CelebrityName'].unique())
        selected_celeb = st.sidebar.selectbox("Select Celebrity", celebrities)
        
        st.header(f"Celebrity: {selected_celeb} (Season {selected_season})")
        
        # Filter for Celeb
        celeb_res = season_results[season_results['CelebrityName'] == selected_celeb].sort_values('Week')
        celeb_orig = season_original[season_original['celebrity_name'] == selected_celeb]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Performance Data")
            st.dataframe(celeb_res, use_container_width=True)
            
        with col2:
            st.subheader("Original Record")
            st.dataframe(celeb_orig, use_container_width=True)
            
        # Stats
        avg_score = celeb_res['JudgeScore'].mean()
        max_score = celeb_res['JudgeScore'].max()
        min_score = celeb_res['JudgeScore'].min()
        std_score = celeb_res['JudgeScore'].std()
        weeks_count = celeb_res['Week'].nunique()
        last_status = celeb_res.sort_values('Week').iloc[-1]['Status']
        
        st.subheader("Key Statistics")
        st_col1, st_col2, st_col3, st_col4 = st.columns(4)
        st_col1.metric("Average Score", f"{avg_score:.2f}")
        st_col2.metric("Max Score", f"{max_score}")
        st_col3.metric("Weeks Competed", f"{weeks_count}")
        st_col4.metric("Latest Status", f"{last_status}")

        # Visualization
        st.subheader("Trajectory Analysis")
        
        # Line Chart: Judge Score
        fig_score = px.line(celeb_res, x='Week', y='JudgeScore', markers=True, title="Judge Score over Weeks")
        fig_score.update_xaxes(dtick=1)
        st.plotly_chart(fig_score, use_container_width=True)
        
        # Visualizing Audience Vote Ranges
        ranges = celeb_res['Possible_Audience_Vote_Range'].apply(parse_range)
        min_vals = [r[0] for r in ranges]
        max_vals = [r[1] for r in ranges]
        
        celeb_res['MinAudScale'] = min_vals
        celeb_res['MaxAudScale'] = max_vals
        
        rule_type = celeb_res['RuleType'].iloc[0]
        y_label = "Audience Rank" if rule_type == 'Rank' else "Audience %"
        
        fig_range = go.Figure()
        
        # Dumbbell Plot
        # Add Lines
        for i, row in celeb_res.iterrows():
            if row['MinAudScale'] is not None:
                # Line connecting Min and Max
                fig_range.add_trace(go.Scatter(
                    x=[row['Week'], row['Week']],
                    y=[row['MinAudScale'], row['MaxAudScale']],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
        # Add Markers for Min and Max
        fig_range.add_trace(go.Scatter(
            x=celeb_res['Week'],
            y=celeb_res['MinAudScale'],
            mode='markers',
            name='Min Likely Vote',
            marker=dict(color='red', size=10, symbol='circle')
        ))
        
        fig_range.add_trace(go.Scatter(
            x=celeb_res['Week'],
            y=celeb_res['MaxAudScale'],
            mode='markers',
            name='Max Likely Vote',
            marker=dict(color='green', size=10, symbol='circle')
        ))

        fig_range.update_layout(
            title=f"Possible Audience Vote {y_label} Ranges",
            xaxis_title="Week",
            yaxis_title=y_label,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_range.update_xaxes(dtick=1)
        
        if rule_type == 'Rank':
            fig_range.update_yaxes(autorange="reversed", dtick=1) # Rank 1 is top, Integer ticks
            
        st.plotly_chart(fig_range, use_container_width=True)

if __name__ == "__main__":
    main()

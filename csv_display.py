import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from pathlib import Path

# Page Config
st.set_page_config(layout="wide", page_title="MCM Results Viewer")

# File Paths
# File Paths
BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / 'MCM_Problem_C_Results.csv'
MC_RESULTS_FILE = BASE_DIR / 'MCM_Problem_C_MonteCarlo_Results.csv'
# Original data might be useful, but let's stick to Results for now as requested for "calculated data"
# We can load original for extra context if needed, but the user focused on the output tables.
# User said "Show two tables", implies Result and Original?
# "Show data from both tables".
ORIGINAL_FILE = BASE_DIR / '2026_MCM_Problem_C_Data.csv'

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"^\\ufeff", "", str(c)).strip() for c in df.columns]
    return df

def _numeric_columns(df: pd.DataFrame, exclude=None):
    exclude = set(exclude or [])
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)
    return numeric_cols

def _is_rank_metric(metric_name: str, rule_type=None) -> bool:
    if metric_name and "rank" in metric_name.lower():
        return True
    if rule_type == "Rank" and metric_name in {"Mean_Rank", "Min_Rank", "Max_Rank", "CI_Lower", "CI_Upper"}:
        return True
    return False

@st.cache_data
def load_data():
    results_df = _normalize_columns(pd.read_csv(RESULTS_FILE))
    original_df = _normalize_columns(pd.read_csv(ORIGINAL_FILE))
    try:
        mc_df = _normalize_columns(pd.read_csv(MC_RESULTS_FILE))
    except Exception as e:
        print(f"Warning: Could not load Monte Carlo file: {e}")
        mc_df = pd.DataFrame() # Return empty if not found to avoid crash
    return results_df, original_df, mc_df

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
        results_df, original_df, mc_df = load_data()
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
    
    season_mc = pd.DataFrame()
    if not mc_df.empty:
        season_mc = mc_df[mc_df['Season'] == selected_season]
    
    # View Mode
    view_mode = st.sidebar.radio("View Mode", ["Season Overview", "Celebrity Detail"])
    
    if view_mode == "Season Overview":
        st.header(f"Season {selected_season} Overview")
        
        # Display Tabs
        tab_labels = ["Calculated Results", "Original Data"]
        if not season_mc.empty:
            tab_labels.append("Monte Carlo Simulations")
            
        tabs = st.tabs(tab_labels)
        
        with tabs[0]:
            st.subheader("Calculated Rankings & Ranges")
            st.dataframe(season_results, use_container_width=True)
            
        with tabs[1]:
            st.subheader("Original Raw Data")
            st.dataframe(season_original, use_container_width=True)
            
        if not season_mc.empty:
            with tabs[2]:
                st.subheader("Monte Carlo Simulation Results")
                st.dataframe(season_mc, use_container_width=True)
                
                st.subheader("Monte Carlo Metric Explorer")
                mc_metric_cols = _numeric_columns(season_mc, exclude=["Season", "Week"])
                if mc_metric_cols:
                    default_metric = "Mean_Rank" if "Mean_Rank" in mc_metric_cols else mc_metric_cols[0]
                    metric_idx = mc_metric_cols.index(default_metric)
                    metric = st.selectbox(
                        "Select Metric",
                        mc_metric_cols,
                        index=metric_idx,
                        key=f"mc_metric_season_{selected_season}",
                    )
                    group_mode = st.radio(
                        "Group By",
                        ["Celebrity", "Week"],
                        horizontal=True,
                        key=f"mc_group_season_{selected_season}",
                    )
                    plot_df = season_mc.copy()
                    plot_df["Week"] = pd.to_numeric(plot_df["Week"], errors="coerce")
                    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

                    if group_mode == "Celebrity" and "CelebrityName" in plot_df.columns:
                        fig_mc_box = px.box(
                            plot_df,
                            x="CelebrityName",
                            y=metric,
                            points="all",
                            title=f"{metric} Distribution by Celebrity",
                        )
                        if _is_rank_metric(metric, None):
                            fig_mc_box.update_layout(yaxis_autorange="reversed")
                        st.plotly_chart(fig_mc_box, use_container_width=True)
                    elif group_mode == "Week":
                        agg = plot_df.groupby("Week", as_index=False)[metric].mean()
                        fig_mc_line = px.line(
                            agg,
                            x="Week",
                            y=metric,
                            markers=True,
                            title=f"Average {metric} by Week",
                        )
                        fig_mc_line.update_xaxes(dtick=1)
                        if _is_rank_metric(metric, None):
                            fig_mc_line.update_layout(yaxis_autorange="reversed")
                        st.plotly_chart(fig_mc_line, use_container_width=True)
                else:
                    st.info("No numeric Monte Carlo metrics found to visualize.")
            
        # Charts
        st.subheader("Performance Trends")
        
        # Average Score Chart
        avg_scores = season_results.groupby('CelebrityName')['JudgeScore'].mean().sort_values()
        fig_avg = px.bar(avg_scores, orientation='h', title="Average Judge Score by Celebrity")
        st.plotly_chart(fig_avg, use_container_width=True)

        st.subheader("Additional Metrics")
        res_metric_cols = _numeric_columns(season_results, exclude=["Season", "Week"])
        if res_metric_cols:
            default_metric = "JudgeScore" if "JudgeScore" in res_metric_cols else res_metric_cols[0]
            metric = st.selectbox(
                "Select Metric",
                res_metric_cols,
                index=res_metric_cols.index(default_metric),
                key=f"res_metric_season_{selected_season}",
            )
            group_mode = st.radio(
                "Group By",
                ["Celebrity", "Week"],
                horizontal=True,
                key=f"res_group_season_{selected_season}",
            )
            plot_df = season_results.copy()
            plot_df["Week"] = pd.to_numeric(plot_df["Week"], errors="coerce")
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

            if group_mode == "Celebrity":
                agg = plot_df.groupby("CelebrityName", as_index=False)[metric].mean().sort_values(metric)
                fig_metric = px.bar(agg, x=metric, y="CelebrityName", orientation="h",
                                    title=f"Average {metric} by Celebrity")
                if _is_rank_metric(metric, None):
                    fig_metric.update_layout(xaxis_autorange="reversed")
                st.plotly_chart(fig_metric, use_container_width=True)
            else:
                agg = plot_df.groupby("Week", as_index=False)[metric].mean()
                fig_metric = px.line(agg, x="Week", y=metric, markers=True,
                                     title=f"Average {metric} by Week")
                fig_metric.update_xaxes(dtick=1)
                if _is_rank_metric(metric, None):
                    fig_metric.update_layout(yaxis_autorange="reversed")
                st.plotly_chart(fig_metric, use_container_width=True)
        else:
            st.info("No numeric calculated metrics found to visualize.")
        
    elif view_mode == "Celebrity Detail":
        celebrities = sorted(season_results['CelebrityName'].unique())
        selected_celeb = st.sidebar.selectbox("Select Celebrity", celebrities)
        
        st.header(f"Celebrity: {selected_celeb} (Season {selected_season})")
        
        # Filter for Celeb
        celeb_res = season_results[season_results['CelebrityName'] == selected_celeb].sort_values('Week')
        celeb_orig = season_original[season_original['celebrity_name'] == selected_celeb]
        celeb_mc = pd.DataFrame()
        if not season_mc.empty:
             celeb_mc = season_mc[season_mc['CelebrityName'] == selected_celeb].sort_values('Week')
        
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
        stats_cols = st.columns(4)
        stats_cols[0].metric("Average Score", f"{avg_score:.2f}")
        stats_cols[1].metric("Max Score", f"{max_score}")
        stats_cols[2].metric("Weeks Competed", f"{weeks_count}")
        stats_cols[3].metric("Latest Status", f"{last_status}")

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

        st.subheader("Additional Calculated Metrics")
        celeb_metric_cols = _numeric_columns(celeb_res, exclude=["Season", "Week"])
        if celeb_metric_cols:
            default_metric = "JudgeScore" if "JudgeScore" in celeb_metric_cols else celeb_metric_cols[0]
            metric = st.selectbox(
                "Select Metric",
                celeb_metric_cols,
                index=celeb_metric_cols.index(default_metric),
                key=f"res_metric_celeb_{selected_season}_{selected_celeb}",
            )
            plot_df = celeb_res.copy()
            plot_df["Week"] = pd.to_numeric(plot_df["Week"], errors="coerce")
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            fig_metric = px.line(
                plot_df,
                x="Week",
                y=metric,
                markers=True,
                title=f"{metric} over Weeks",
            )
            fig_metric.update_xaxes(dtick=1)
            if _is_rank_metric(metric, rule_type):
                fig_metric.update_layout(yaxis_autorange="reversed")
            st.plotly_chart(fig_metric, use_container_width=True)
        else:
            st.info("No numeric calculated metrics found to visualize.")

        if not celeb_mc.empty:
            st.divider()
            st.subheader("Monte Carlo Metrics")
            mc_metric_cols = _numeric_columns(celeb_mc, exclude=["Season", "Week"])
            if mc_metric_cols:
                default_metric = "Mean_Rank" if "Mean_Rank" in mc_metric_cols else mc_metric_cols[0]
                metric = st.selectbox(
                    "Select Metric",
                    mc_metric_cols,
                    index=mc_metric_cols.index(default_metric),
                    key=f"mc_metric_celeb_{selected_season}_{selected_celeb}",
                )
                plot_df = celeb_mc.copy()
                plot_df["Week"] = pd.to_numeric(plot_df["Week"], errors="coerce")
                plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
                rule_type_mc = None
                if "RuleType" in plot_df.columns and plot_df["RuleType"].notna().any():
                    rule_type_mc = plot_df["RuleType"].dropna().iloc[0]

                fig_mc = go.Figure()

                if metric == "Mean_Rank" and {"CI_Lower", "CI_Upper"}.issubset(plot_df.columns):
                    plot_df["CI_Lower"] = pd.to_numeric(plot_df["CI_Lower"], errors="coerce")
                    plot_df["CI_Upper"] = pd.to_numeric(plot_df["CI_Upper"], errors="coerce")
                    fig_mc.add_trace(go.Scatter(
                        x=plot_df["Week"],
                        y=plot_df[metric],
                        mode="lines+markers",
                        name=f"{metric}",
                        line=dict(color="purple", width=3),
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=plot_df["CI_Upper"] - plot_df[metric],
                            arrayminus=plot_df[metric] - plot_df["CI_Lower"],
                            visible=True,
                            color="purple",
                            thickness=1.5,
                            width=4,
                        ),
                    ))
                else:
                    fig_mc.add_trace(go.Scatter(
                        x=plot_df["Week"],
                        y=plot_df[metric],
                        mode="lines+markers",
                        name=metric,
                        line=dict(color="purple", width=3),
                    ))

                if metric == "Mean_Rank" and {"Min_Rank", "Max_Rank"}.issubset(plot_df.columns):
                    plot_df["Min_Rank"] = pd.to_numeric(plot_df["Min_Rank"], errors="coerce")
                    plot_df["Max_Rank"] = pd.to_numeric(plot_df["Max_Rank"], errors="coerce")
                    fig_mc.add_trace(go.Scatter(
                        x=plot_df["Week"],
                        y=plot_df["Min_Rank"],
                        mode="lines",
                        line=dict(color="rgba(120, 120, 120, 0.5)", width=1),
                        showlegend=False,
                    ))
                    fig_mc.add_trace(go.Scatter(
                        x=plot_df["Week"],
                        y=plot_df["Max_Rank"],
                        mode="lines",
                        fill="tonexty",
                        fillcolor="rgba(120, 120, 120, 0.15)",
                        line=dict(color="rgba(120, 120, 120, 0.5)", width=1),
                        name="Min/Max Rank Range",
                    ))

                fig_mc.update_layout(
                    title=f"Monte Carlo {metric} by Week",
                    xaxis_title="Week",
                    yaxis_title=metric,
                    showlegend=True,
                )
                fig_mc.update_xaxes(dtick=1)
                if _is_rank_metric(metric, rule_type_mc):
                    fig_mc.update_yaxes(autorange="reversed", dtick=1)

                st.plotly_chart(fig_mc, use_container_width=True)
            else:
                st.info("No numeric Monte Carlo metrics found to visualize.")
            
            st.markdown("### Detailed Monte Carlo Data")
            st.dataframe(celeb_mc, use_container_width=True)

if __name__ == "__main__":
    main()

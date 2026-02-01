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
MC_RESULTS_FILE = BASE_DIR / 'MCM_Problem_C_Results_20260201_1748.csv'
DIFF_SUMMARY_FILE = BASE_DIR / 'rank_percent_diff_summary.csv'
RANK_PERCENT_DIFF_SUMMARY_FILE = BASE_DIR / 'MCM_Problem_C_Results_20260201_1748_uncertainty.csv'
ATTRACT_OUTPUT_DIR = BASE_DIR / 'Attract' / 'output'
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

def _coerce_numeric(df: pd.DataFrame, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _is_rank_metric(metric_name: str, rule_type=None) -> bool:
    if metric_name and "rank" in metric_name.lower():
        return True
    if rule_type == "Rank" and metric_name in {"Mean_Rank", "Min_Rank", "Max_Rank", "CI_Lower", "CI_Upper"}:
        return True
    return False

def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return _normalize_columns(pd.read_csv(path))
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
    return pd.DataFrame()

def _list_attract_tables() -> list:
    if not ATTRACT_OUTPUT_DIR.exists():
        return []
    return sorted(ATTRACT_OUTPUT_DIR.rglob("*.csv"))

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

@st.cache_data
def load_extra_tables():
    tables = {}
    for path in [DIFF_SUMMARY_FILE, RANK_PERCENT_DIFF_SUMMARY_FILE]:
        df = _safe_read_csv(path)
        if not df.empty:
            tables[path.stem] = df

    for path in _list_attract_tables():
        df = _safe_read_csv(path)
        if not df.empty:
            rel = path.relative_to(ATTRACT_OUTPUT_DIR)
            tables[f"Attract/{rel.as_posix()}"] = df
    return tables

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

def _chart_config():
    return {
        "displayModeBar": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "plot",
            "height": 900,
            "width": 1400,
            "scale": 2,
        },
    }

def _apply_paper_style(fig, title=None):
    fig.update_layout(
        template="simple_white",
        title=title or fig.layout.title.text,
        font=dict(size=14),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

def _render_table_preview(name: str, df: pd.DataFrame, selected_season=None):
    with st.expander(f"{name} ({len(df)} rows × {len(df.columns)} cols)", expanded=False):
        view_df = df.copy()
        with st.container():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                search_text = st.text_input(
                    "Search (any column contains)",
                    value="",
                    key=f"search_{name}",
                )
            with c2:
                max_rows = st.number_input(
                    "Max rows",
                    min_value=50,
                    max_value=5000,
                    value=500,
                    step=50,
                    key=f"maxrows_{name}",
                )
            with c3:
                show_cols = st.multiselect(
                    "Columns",
                    list(view_df.columns),
                    default=list(view_df.columns),
                    key=f"cols_{name}",
                )
        if show_cols:
            view_df = view_df[show_cols]
        if selected_season is not None and "Season" in view_df.columns:
            if st.checkbox(f"Filter Season = {selected_season}", key=f"filter_{name}"):
                view_df = view_df[view_df["Season"] == selected_season]
        if search_text:
            mask = pd.Series(False, index=view_df.index)
            for col in view_df.columns:
                mask = mask | view_df[col].astype(str).str.contains(search_text, case=False, na=False)
            view_df = view_df[mask]
        if max_rows and len(view_df) > max_rows:
            view_df = view_df.head(int(max_rows))
        st.dataframe(view_df, use_container_width=True, height=420)
        csv_bytes = view_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=f"{name.replace('/', '_')}.csv",
            mime="text/csv",
            key=f"dl_{name}",
        )

def _render_overlay_single_table(df: pd.DataFrame, title_prefix: str = ""):
    if df.empty:
        st.info("No data available.")
        return
    cols = list(df.columns)
    numeric_cols = _numeric_columns(df)
    if not numeric_cols:
        st.info("No numeric columns to plot.")
        return
    x_default = "Week" if "Week" in cols else ("Season" if "Season" in cols else cols[0])
    x_col = st.selectbox("X-axis", cols, index=cols.index(x_default))
    y_cols = st.multiselect(
        "Y-axis (overlay)",
        numeric_cols,
        default=[numeric_cols[0]],
    )
    color_col = st.selectbox("Color / Group (optional)", ["<None>"] + cols)
    chart_type = st.radio("Chart Type", ["Line", "Bar", "Scatter"], horizontal=True)
    agg_mode = st.selectbox("Aggregate duplicates (optional)", ["none", "mean", "sum", "median", "min", "max"])
    paper_style = st.checkbox("Paper-style (clean)", value=True)

    plot_df = df.copy()
    plot_df = _coerce_numeric(plot_df, y_cols)
    x_numeric = pd.to_numeric(plot_df[x_col], errors="coerce")
    if x_numeric.notna().any():
        plot_df[x_col] = x_numeric

    if agg_mode != "none":
        group_cols = [x_col] + ([color_col] if color_col != "<None>" else [])
        plot_df = plot_df.groupby(group_cols, as_index=False)[y_cols].agg(agg_mode)

    fig = go.Figure()
    if color_col != "<None>":
        groups = plot_df[color_col].dropna().unique().tolist()
        for grp in groups:
            gdf = plot_df[plot_df[color_col] == grp]
            for y in y_cols:
                name = f"{grp} · {y}" if len(y_cols) > 1 else str(grp)
                if chart_type == "Bar":
                    fig.add_trace(go.Bar(x=gdf[x_col], y=gdf[y], name=name))
                elif chart_type == "Scatter":
                    fig.add_trace(go.Scatter(x=gdf[x_col], y=gdf[y], mode="markers", name=name))
                else:
                    fig.add_trace(go.Scatter(x=gdf[x_col], y=gdf[y], mode="lines+markers", name=name))
    else:
        for y in y_cols:
            if chart_type == "Bar":
                fig.add_trace(go.Bar(x=plot_df[x_col], y=plot_df[y], name=y))
            elif chart_type == "Scatter":
                fig.add_trace(go.Scatter(x=plot_df[x_col], y=plot_df[y], mode="markers", name=y))
            else:
                fig.add_trace(go.Scatter(x=plot_df[x_col], y=plot_df[y], mode="lines+markers", name=y))

    fig.update_layout(
        title=f"{title_prefix}{x_col} vs {', '.join(y_cols)}",
        xaxis_title=x_col,
        yaxis_title=", ".join(y_cols),
    )
    if paper_style:
        _apply_paper_style(fig)
    st.plotly_chart(fig, use_container_width=True, config=_chart_config())

    html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    st.download_button(
        "Download Chart (HTML)",
        data=html.encode("utf-8"),
        file_name="chart.html",
        mime="text/html",
    )

def _render_overlay_multi_tables(table_map: dict, selected_tables: list):
    dfs = [table_map[name] for name in selected_tables]
    if not dfs:
        return
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)
    common_cols = sorted(list(common_cols))
    if not common_cols:
        st.info("Selected tables have no common columns to align on.")
        return

    common_numeric = set(_numeric_columns(dfs[0]))
    for df in dfs[1:]:
        common_numeric &= set(_numeric_columns(df))
    common_numeric = sorted(list(common_numeric))
    if not common_numeric:
        st.info("Selected tables have no common numeric columns to overlay.")
        return

    x_default = "Week" if "Week" in common_cols else ("Season" if "Season" in common_cols else common_cols[0])
    x_col = st.selectbox("X-axis (common)", common_cols, index=common_cols.index(x_default))
    y_col = st.selectbox("Y-axis (common numeric)", common_numeric, index=0)
    agg = st.selectbox("Aggregation", ["mean", "sum", "median", "min", "max"])
    chart_type = st.radio("Chart Type", ["Line", "Bar", "Scatter"], horizontal=True, key="multi_chart_type")
    paper_style = st.checkbox("Paper-style (clean)", value=True, key="multi_paper")

    plot_frames = []
    for name in selected_tables:
        df = table_map[name]
        temp = df.copy()
        temp = _coerce_numeric(temp, [y_col])
        x_numeric = pd.to_numeric(temp[x_col], errors="coerce")
        if x_numeric.notna().any():
            temp[x_col] = x_numeric
        temp = temp.dropna(subset=[x_col, y_col])
        if temp.empty:
            continue
        if agg:
            temp = temp.groupby(x_col, as_index=False)[y_col].agg(agg)
        temp["Source"] = name
        plot_frames.append(temp)

    if not plot_frames:
        st.info("No data to plot after aggregation.")
        return

    plot_df = pd.concat(plot_frames, ignore_index=True)
    if chart_type == "Bar":
        fig = px.bar(plot_df, x=x_col, y=y_col, color="Source", barmode="group")
    elif chart_type == "Scatter":
        fig = px.scatter(plot_df, x=x_col, y=y_col, color="Source")
    else:
        fig = px.line(plot_df, x=x_col, y=y_col, color="Source", markers=True)
    fig.update_layout(
        title=f"Overlay: {y_col} by {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    if paper_style:
        _apply_paper_style(fig)
    st.plotly_chart(fig, use_container_width=True, config=_chart_config())

    html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    st.download_button(
        "Download Chart (HTML)",
        data=html.encode("utf-8"),
        file_name="overlay_chart.html",
        mime="text/html",
        key="dl_overlay_chart",
    )

def _render_variable_guide(
    results_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    original_df: pd.DataFrame,
    extra_tables: dict,
):
    st.caption("不同表格的字段含义如下（根据当前数据列动态展示）。")

    results_desc = {
        "CelebrityName": "选手姓名",
        "Season": "赛季编号",
        "Week": "周次",
        "RuleType": "规则类型（Rank=观众投票按排名，Percent=观众投票按百分比）",
        "JudgeScore": "该周评委总分",
        "JudgeScore_Normalization": "评委分数的规范化结果（排名/百分比）",
        "Possible_Audience_Vote_Range": "观众投票可能范围（排名区间或百分比区间）",
        "Predicted_Audience_Percent": "模型预测的观众投票百分比",
        "Predicted_Audience_Rank": "模型预测的观众投票排名",
        "Loss_Total": "优化总损失",
        "Loss_Constraint": "约束损失",
        "Loss_Smooth": "平滑损失",
        "Loss_Corr": "相关性损失",
        "Loss_Reg": "正则化损失",
        "Loss_Diversity": "多样性损失",
        "Loss_Trend": "搜索热度损失",
        "Status": "当周状态（Safe/Eliminated 等）",
    }

    mc_desc = {
        "Season": "赛季编号",
        "Week": "周次",
        "CelebrityName": "选手姓名",
        "Mean_Rank": "蒙特卡洛估计的平均排名",
        "Min_Rank": "模拟中的最小排名",
        "Max_Rank": "模拟中的最大排名",
        "CI_Lower": "排名置信区间下界",
        "CI_Upper": "排名置信区间上界",
        "Status": "当周状态（Safe/Eliminated 等）",
        "RuleType": "规则类型（Rank/Percent）",
    }

    original_desc = {
        "celebrity_name": "选手姓名",
        "ballroom_partner": "舞伴姓名",
        "celebrity_industry": "选手行业/职业",
        "celebrity_homestate": "选手家乡州/省",
        "celebrity_homecountry/region": "选手家乡国家/地区",
        "celebrity_age_during_season": "赛季期间年龄",
        "season": "赛季编号",
        "results": "赛季最终结果/名次描述",
        "placement": "赛季最终名次（数字）",
    }

    diff_desc = {
        "Season": "赛季编号",
        "Week": "周次",
        "Participants": "当周参赛人数",
        "Elimination_Event": "是否发生淘汰（0/1）",
        "Elim_Percent": "Percent 规则下预测的淘汰者",
        "Elim_Rank": "Rank 规则下预测的淘汰者",
        "Elim_Changed": "两种规则下淘汰者是否不同（0/1）",
        "Rank_Diff_Mean": "两种规则下名次差异的平均值",
        "Diff_Score_Normalized": "标准化后的差异分数（越大代表差异越明显）",
    }

    rank_percent_diff_desc = {
        "CelebrityName": "选手姓名",
        "Season": "赛季编号",
        "Week": "周次",
        "RuleType": "规则类型（Rank/Percent）",
        "Possible_Audience_Vote_Range": "观众投票可能范围（排名区间或百分比区间）",
        "Uncertainty_Range_Min": "不确定性区间下界",
        "Uncertainty_Range_Max": "不确定性区间上界",
        "Uncertainty_Range_Width": "不确定性区间宽度（Max-Min）",
        "Uncertainty_Index": "不确定性指数（归一化宽度）",
    }

    attract_common_desc = {
        "RuleType": "规则类型（Rank/Percent）",
        "Value": "分组取值（行业/姓名/地区等）",
        "Target": "目标指标（Judge/Audience）",
        "N": "样本量",
        "Mean_Judge_Norm": "评委规范化均值",
        "Mean_Audience_Norm": "观众规范化均值",
        "Judge_Lift": "评委相对整体的提升值",
        "Audience_Lift": "观众相对整体的提升值",
        "Judge_Std": "评委规范化标准差",
        "Audience_Std": "观众规范化标准差",
        "Slope": "线性回归斜率",
        "Intercept": "线性回归截距",
        "Correlation": "相关系数",
    }

    def render_table(df: pd.DataFrame, desc_map: dict, title: str):
        st.subheader(title)
        if df.empty:
            st.info("暂无数据可展示。")
            return
        rows = []
        for col in df.columns:
            if col in desc_map:
                desc = desc_map[col]
            else:
                m = re.match(r"week(\\d+)_judge(\\d+)_score", str(col))
                if m:
                    desc = f"第{m.group(1)}周评委{m.group(2)}打分"
                else:
                    desc = "（新增字段）请根据数据含义补充说明"
            rows.append({"字段": col, "说明": desc})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    tabs = st.tabs(["Calculated Results", "Monte Carlo", "Original Data", "Diff Summary", "Uncertainty Summary", "Attract Tables"])
    with tabs[0]:
        render_table(results_df, results_desc, "Calculated Results 字段")
    with tabs[1]:
        render_table(mc_df, mc_desc, "Monte Carlo 字段")
    with tabs[2]:
        render_table(original_df, original_desc, "Original Data 字段")
    with tabs[3]:
        diff_df = extra_tables.get("rank_percent_diff_summary", pd.DataFrame())
        render_table(diff_df, diff_desc, "rank_percent_diff_summary 字段")
    with tabs[4]:
        rp_df = extra_tables.get("MCM_Problem_C_Results_20260201_1748_uncertainty", pd.DataFrame())
        render_table(rp_df, rank_percent_diff_desc, "uncertainty 字段")
    with tabs[5]:
        attract_tables = {
            k: v for k, v in extra_tables.items() if k.startswith("Attract/")
        }
        if not attract_tables:
            st.info("暂无 Attract 表格。")
        else:
            for name, df in attract_tables.items():
                render_table(df, attract_common_desc, f"{name} 字段")

def main():
    st.title("MCM Problem C - Analysis Dashboard")
    
    try:
        results_df, original_df, mc_df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    extra_tables = load_extra_tables()

    with st.expander("变量说明", expanded=False):
        _render_variable_guide(results_df, mc_df, original_df, extra_tables)

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
    view_mode = st.sidebar.radio("View Mode", ["Season Overview", "Celebrity Detail", "Table Explorer"])
    
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
    else:
        st.header("Table Explorer")
        all_tables = {
            "Results (Calculated)": results_df,
            "Original Data": original_df,
        }
        if not mc_df.empty:
            all_tables["Monte Carlo"] = mc_df
        all_tables.update(extra_tables)

        if not all_tables:
            st.info("No tables available.")
            return

        table_names = sorted(all_tables.keys())
        default_tables = [
            name
            for name in table_names
            if name
            in {
                "rank_percent_diff_summary",
                "MCM_Problem_C_Results_20260201_1748_uncertainty",
            }
        ]
        if not default_tables:
            default_tables = [table_names[0]]

        selected_tables = st.multiselect(
            "Select tables to view / overlay",
            table_names,
            default=default_tables,
        )

        st.subheader("Table Previews")
        for name in selected_tables:
            _render_table_preview(name, all_tables[name], selected_season=selected_season)

        st.subheader("Overlay Charts")
        if len(selected_tables) == 1:
            _render_overlay_single_table(all_tables[selected_tables[0]], title_prefix=f"{selected_tables[0]}: ")
        elif len(selected_tables) > 1:
            _render_overlay_multi_tables(all_tables, selected_tables)
        else:
            st.info("Select at least one table to plot.")

        st.subheader("Quick Charts (Paper-ready presets)")
        if "rank_percent_diff_summary" in all_tables:
            st.markdown("**rank_percent_diff_summary**")
            diff_df = all_tables["rank_percent_diff_summary"].copy()
            diff_df = _coerce_numeric(diff_df, ["Week", "Season", "Diff_Score_Normalized"])
            if {"Week", "Diff_Score_Normalized"}.issubset(diff_df.columns):
                fig = px.line(
                    diff_df.groupby("Week", as_index=False)["Diff_Score_Normalized"].mean(),
                    x="Week",
                    y="Diff_Score_Normalized",
                    markers=True,
                    title="Avg Diff_Score_Normalized by Week",
                )
                _apply_paper_style(fig)
                st.plotly_chart(fig, use_container_width=True, config=_chart_config())

        if "MCM_Problem_C_Results_20260201_1748_uncertainty" in all_tables:
            st.markdown("**uncertainty**")
            rp_df = all_tables["MCM_Problem_C_Results_20260201_1748_uncertainty"].copy()
            rp_df = _coerce_numeric(rp_df, ["Week", "Season", "Uncertainty_Index"])
            if {"Week", "Uncertainty_Index"}.issubset(rp_df.columns):
                fig = px.line(
                    rp_df.groupby("Week", as_index=False)["Uncertainty_Index"].mean(),
                    x="Week",
                    y="Uncertainty_Index",
                    markers=True,
                    title="Avg Uncertainty_Index by Week",
                )
                _apply_paper_style(fig)
                st.plotly_chart(fig, use_container_width=True, config=_chart_config())

        attract_keys = [k for k in all_tables.keys() if k.startswith("Attract/")]
        if attract_keys:
            st.markdown("**Attract (effects)**")
            default_key = "Attract/effects_celebrity_industry.csv"
            select_key = st.selectbox(
                "Select Attract table",
                attract_keys,
                index=attract_keys.index(default_key) if default_key in attract_keys else 0,
            )
            attr_df = all_tables[select_key].copy()
            numeric_cols = _numeric_columns(attr_df)
            x_col = "Value" if "Value" in attr_df.columns else attr_df.columns[0]
            y_col = "Audience_Lift" if "Audience_Lift" in numeric_cols else (numeric_cols[0] if numeric_cols else None)
            if y_col is not None and x_col in attr_df.columns:
                fig = px.bar(
                    attr_df.sort_values(y_col, ascending=False).head(20),
                    x=y_col,
                    y=x_col,
                    orientation="h",
                    title=f"Top 20 {x_col} by {y_col}",
                )
                _apply_paper_style(fig)
                st.plotly_chart(fig, use_container_width=True, config=_chart_config())

if __name__ == "__main__":
    main()

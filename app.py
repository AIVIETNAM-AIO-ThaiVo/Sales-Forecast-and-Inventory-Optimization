from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inventory_engine import Costs, available_series, forecast_series, grid_search, load_dataset, simulate


ROOT = Path(__file__).parent
OPT_START, OPT_END = 1830, 1913
TEST_START, TEST_END = 1914, 1941
WARMUP_DAYS = 28

st.set_page_config(page_title="Inventory Control Tower", page_icon="📦", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container {max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: white; border: 1px solid #e7ebf2; border-radius: 12px; padding: 14px 16px;}
    .hero {background: linear-gradient(135deg,#152b49 0%,#275d76 100%); color:white; padding: 26px 30px; border-radius: 16px; margin-bottom: 22px;}
    .hero h1 {margin:0; font-size: 2rem; letter-spacing: -.02em;}
    .hero p {margin:8px 0 0; color:#d6e5ef; font-size:1rem;}
    .helper {background:#f5f7fb; border:1px solid #e6eaf0; padding:12px 16px; border-radius:10px; margin:8px 0 18px; color:#536176;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_dataset():
    return load_dataset(ROOT)


ds = get_dataset()
series = available_series(ds).copy()
series["item"] = series["id"].str.extract(r"HOBBIES_[12]_([^_]+)")[0].fillna(series["id"])


def selector(prefix: str):
    """Independent cascading Store -> Department -> Item selector for one tab."""
    stores = sorted(series["store"].unique())
    store = st.selectbox("Store", stores, key=f"{prefix}_store")
    departments = sorted(series.loc[series["store"] == store, "department"].unique())
    dept = st.selectbox("Department", departments, key=f"{prefix}_dept")
    items = sorted(series.loc[(series["store"] == store) & (series["department"] == dept), "item"].unique())
    item = st.selectbox("Item / SKU", items, key=f"{prefix}_item")
    selected_id = series.loc[(series["store"] == store) & (series["department"] == dept) & (series["item"] == item), "id"].iloc[0]
    return selected_id, store, dept, item


def selected_frame(selected_id: str, start: int = OPT_START - WARMUP_DAYS, end: int = TEST_END) -> pd.DataFrame:
    frame = ds.frame[(ds.frame["__id"] == selected_id) & (ds.frame["__d"].between(start, end))].sort_values("__d").copy()
    if "__date" not in frame.columns:
        frame["__date"] = pd.NaT
    return frame


def attach_lead_forecast(frame: pd.DataFrame, lead_time: int) -> pd.DataFrame:
    """Build the rolling forecast total consumed by policy at each decision day."""
    frame = frame.sort_values("__d").copy()
    values = frame["__forecast"].to_numpy(dtype=float)
    frame["__lead_forecast"] = [float(values[i + 1 : i + 1 + lead_time].sum()) for i in range(len(values))]
    return frame


def price_default(frame: pd.DataFrame) -> float:
    prices = pd.to_numeric(frame.loc[frame["__d"].between(OPT_START, OPT_END), "sell_price"], errors="coerce").dropna()
    return float(prices.median()) if not prices.empty else 10.0


def cost_controls(prefix: str, frame: pd.DataFrame):
    default_price = price_default(frame)
    c1, c2, c3, c4, c5 = st.columns(5)
    reference_price = c1.number_input("Reference price ($)", min_value=0.01, value=round(default_price, 2), step=0.1, key=f"{prefix}_price")
    cost_ratio = c2.number_input("Unit cost ratio", min_value=0.0, max_value=1.0, value=0.76, step=0.01, key=f"{prefix}_ratio")
    holding_rate = c3.number_input("Annual holding rate", min_value=0.0, value=0.25, step=0.01, key=f"{prefix}_holding")
    ordering_cost = c4.number_input("Ordering cost K ($)", min_value=0.0, value=5.0, step=0.5, key=f"{prefix}_ordering")
    penalty = c5.number_input("Stockout penalty p ($)", min_value=0.0, value=round(default_price * (1 - cost_ratio), 2), step=0.1, key=f"{prefix}_penalty")
    return Costs(reference_price, cost_ratio, holding_rate, ordering_cost, penalty)


def run_backtest(selected_id: str, costs: Costs, lead_time: int, initial_mode: str, initial_manual: float, alphas: list[float], betas: list[float], fill_target: float):
    frame = selected_frame(selected_id)
    frame["__forecast"], _ = forecast_series(ds, selected_id, frame)
    frame = attach_lead_forecast(frame, lead_time)
    opt = frame[frame["__d"].between(OPT_START, OPT_END)].copy()
    test = frame[frame["__d"].between(TEST_START, TEST_END)].copy()
    initial = initial_manual if initial_mode == "Manual" else float(opt["__demand"].mean() * (lead_time + 1))
    grid = grid_search(opt, lead_time, initial, costs, alphas, betas, fill_target or None)
    feasible = grid[grid["feasible"]]
    best = (feasible if not feasible.empty else grid).sort_values("total_cost").iloc[0]
    runs = {}
    for name, alpha, beta in [("Historical baseline", 1.0, 1.0), ("Forecast-based", 1.0, 1.0), ("Forecast + alpha-beta", float(best.alpha), float(best.beta))]:
        runs[name] = simulate(test, name, lead_time, initial, costs, alpha, beta)
    summary = pd.DataFrame([{"Strategy": name, **metrics} for name, (_, metrics) in runs.items()])
    return frame, opt, test, initial, grid, best, runs, summary


def batch_strategy_statistics(candidate_rows: pd.DataFrame, lead_time: int, alphas: list[float], betas: list[float], progress=None) -> pd.DataFrame:
    """Find item-level alpha/beta on optimization, then assign lowest test-cost strategy."""
    records = []
    total = len(candidate_rows)
    for index, (_, item_row) in enumerate(candidate_rows.iterrows(), start=1):
        selected_id = item_row["id"]
        frame = selected_frame(selected_id)
        frame["__forecast"], _ = forecast_series(ds, selected_id, frame)
        frame = attach_lead_forecast(frame, lead_time)
        opt = frame[frame["__d"].between(OPT_START, OPT_END)]
        test = frame[frame["__d"].between(TEST_START, TEST_END)]
        reference_price = price_default(frame)
        costs = Costs(reference_price, 0.76, 0.25, 5.0, reference_price * 0.24)
        initial = float(opt["__demand"].mean() * (lead_time + 1))
        result = {}
        _, result["Historical baseline"] = simulate(test, "Historical baseline", lead_time, initial, costs, 1.0, 1.0)
        _, result["Forecast-based"] = simulate(test, "Forecast-based", lead_time, initial, costs, 1.0, 1.0)
        item_grid = grid_search(opt, lead_time, initial, costs, alphas, betas)
        best_item = item_grid.sort_values("total_cost").iloc[0]
        _, result["Forecast + alpha-beta"] = simulate(test, "Forecast + alpha-beta", lead_time, initial, costs, float(best_item["alpha"]), float(best_item["beta"]))
        winner = min(result, key=lambda name: result[name]["total_cost"])
        records.append({
            "store_id": item_row["store"],
            "department": item_row["department"],
            "item_id": item_row["item"],
            "series_id": selected_id,
            "recommended_strategy": winner,
            "historical_cost": result["Historical baseline"]["total_cost"],
            "forecast_cost": result["Forecast-based"]["total_cost"],
            "alpha_beta_cost": result["Forecast + alpha-beta"]["total_cost"],
            "historical_shortage_rate": 1.0 - result["Historical baseline"]["fill_rate"],
            "forecast_shortage_rate": 1.0 - result["Forecast-based"]["fill_rate"],
            "alpha_beta_shortage_rate": 1.0 - result["Forecast + alpha-beta"]["fill_rate"],
            "historical_average_inventory": result["Historical baseline"]["average_inventory"],
            "forecast_average_inventory": result["Forecast-based"]["average_inventory"],
            "alpha_beta_average_inventory": result["Forecast + alpha-beta"]["average_inventory"],
            "recommended_total_cost": result[winner]["total_cost"],
            "recommended_fill_rate": result[winner]["fill_rate"],
            "shortage_rate": 1.0 - result[winner]["fill_rate"],
            "average_inventory": result[winner]["average_inventory"],
            "best_alpha": float(best_item["alpha"]),
            "best_beta": float(best_item["beta"]),
        })
        if progress is not None and (index == total or index % 10 == 0):
            progress.progress(index / max(total, 1), text=f"Evaluated {index}/{total} items")
    return pd.DataFrame(records)


def all_item_benchmark(stats: pd.DataFrame) -> pd.DataFrame:
    """Average each policy over the same complete item universe."""
    definitions = [
        ("Historical baseline", "historical_cost", "historical_shortage_rate", "historical_average_inventory"),
        ("Forecast-based", "forecast_cost", "forecast_shortage_rate", "forecast_average_inventory"),
        ("Forecast + alpha-beta", "alpha_beta_cost", "alpha_beta_shortage_rate", "alpha_beta_average_inventory"),
    ]
    return pd.DataFrame([
        {
            "Policy": policy,
            "item_count": len(stats),
            "mean_total_cost": stats[cost_col].mean(),
            "mean_shortage_rate": stats[shortage_col].mean(),
            "mean_average_inventory": stats[inventory_col].mean(),
        }
        for policy, cost_col, shortage_col, inventory_col in definitions
    ])


def policy_controls(prefix: str):
    c1, c2, c3 = st.columns(3)
    strategy = c1.selectbox("Policy", ["Historical baseline", "Forecast-based", "Forecast + alpha-beta"], key=f"{prefix}_strategy")
    lead_time = c2.number_input("Lead time (days)", min_value=1, max_value=60, value=10, step=1, key=f"{prefix}_lead")
    initial_mode = c3.selectbox("Initial inventory", ["Auto", "Manual"], key=f"{prefix}_initial_mode")
    initial_manual = st.number_input("Manual units", min_value=0.0, value=25.0, step=1.0, disabled=initial_mode != "Manual", key=f"{prefix}_initial_manual")
    return strategy, int(lead_time), initial_mode, initial_manual


def grid_controls(prefix: str):
    with st.expander("Alpha-beta candidates", expanded=False):
        alpha_text = st.text_input("Alpha", "0.8, 0.9, 1.0, 1.1, 1.2", key=f"{prefix}_alpha")
        beta_text = st.text_input("Beta", "0.5, 0.75, 1.0, 1.25, 1.5", key=f"{prefix}_beta")
    alphas = [float(x.strip()) for x in alpha_text.split(",") if x.strip()]
    betas = [float(x.strip()) for x in beta_text.split(",") if x.strip()]
    return alphas, betas


st.markdown('<div class="hero"><h1>Inventory Control Tower</h1><p>M5 HOBBIES · explore data, forecast and inventory policies in separate workspaces</p></div>', unsafe_allow_html=True)
st.markdown('<div class="helper">Each tab is independent. Choose Store → Department → Item inside the tab, configure only what that analysis needs, then press its Run button.</div>', unsafe_allow_html=True)

tabs = st.tabs(["Data Explorer", "Forecast Explorer", "Inventory Simulation", "Strategy Comparison", "Optimization", "Strategy Statistics", "Strategy Visualization"])

with tabs[0]:
    st.header("Data Explorer")
    st.caption("Explore sales and selling price for one item. This tab does not run inventory simulation.")
    with st.form("data_explorer_form"):
        selected_id, store, dept, item = selector("data")
        d_range = st.slider("Data window", 700, 1941, (OPT_START, TEST_END), key="data_range")
        explore = st.form_submit_button("Run Data Explorer", type="primary")
    if explore:
        st.session_state.data_result = (selected_id, d_range)
    if "data_result" in st.session_state:
        selected_id, d_range = st.session_state.data_result
        frame = selected_frame(selected_id, d_range[0], d_range[1])
        view = frame[["__d", "__demand", "sell_price"]].rename(columns={"__d": "d", "__demand": "sales"}).set_index("d")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean daily sales", f"{view['sales'].mean():.2f}")
        m2.metric("Sales standard deviation", f"{view['sales'].std():.2f}")
        m3.metric("Zero-sales rate", f"{(view['sales'] == 0).mean():.1%}")
        st.line_chart(view, height=360)
        st.dataframe(view.reset_index(), use_container_width=True, hide_index=True)
    else:
        st.info("Configure the selectors above and press Run Data Explorer.")

with tabs[1]:
    st.header("Forecast Explorer")
    st.caption("Inspect actual demand versus LightGBM forecast across the fixed optimization and test windows.")
    with st.form("forecast_explorer_form"):
        selected_id, store, dept, item = selector("forecast")
        run_forecast = st.form_submit_button("Run Forecast Explorer", type="primary")
    if run_forecast:
        frame = selected_frame(selected_id)
        frame["__forecast"], _ = forecast_series(ds, selected_id, frame)
        st.session_state.forecast_result = (selected_id, frame)
    if "forecast_result" in st.session_state:
        selected_id, frame = st.session_state.forecast_result
        opt = frame[frame["__d"].between(OPT_START, OPT_END)]
        view = frame[["__d", "__demand", "__forecast"]].rename(columns={"__d": "d", "__demand": "actual", "__forecast": "forecast"}).set_index("d")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimization bias", f"{(opt['__forecast'] - opt['__demand']).mean():.2f}")
        m2.metric("Optimization MAE", f"{(opt['__forecast'] - opt['__demand']).abs().mean():.2f}")
        m3.metric("Test days", str(TEST_END - TEST_START + 1))
        st.line_chart(view, height=380)
        st.dataframe(view.reset_index(), use_container_width=True, hide_index=True)
    else:
        st.info("Configure the selectors above and press Run Forecast Explorer.")

with tabs[2]:
    st.header("Inventory Simulation")
    st.caption(f"Fixed windows: optimization d={OPT_START}–{OPT_END}; final test d={TEST_START}–{TEST_END}.")
    with st.form("inventory_simulation_form"):
        selected_id, store, dept, item = selector("simulation")
        strategy, lead_time, initial_mode, initial_manual = policy_controls("simulation")
        costs = cost_controls("simulation", selected_frame(selected_id))
        fill_target = st.number_input("Minimum fill rate (0 = off)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="simulation_fill")
        alphas, betas = grid_controls("simulation")
        run_simulation = st.form_submit_button("Run Inventory Simulation", type="primary")
    if run_simulation:
        st.session_state.simulation_result = run_backtest(selected_id, costs, lead_time, initial_mode, initial_manual, alphas, betas, fill_target)
    if "simulation_result" in st.session_state:
        frame, opt, test, initial, grid, best, runs, summary = st.session_state.simulation_result
        daily, metrics = runs[strategy]
        m = st.columns(5)
        m[0].metric("Fill rate", f"{metrics['fill_rate']:.1%}")
        m[1].metric("Total cost", f"${metrics['total_cost']:,.2f}")
        m[2].metric("Orders", str(metrics["orders"]))
        m[3].metric("Average inventory", f"{metrics['average_inventory']:.1f}")
        m[4].metric("Lost sales", f"{metrics['lost_sales']:.0f}")
        st.line_chart(daily.set_index("d")[["on_hand", "inventory_position", "rop"]], height=330)
        st.line_chart(daily.set_index("d")[["demand", "fulfilled", "lost_sales"]], height=280)
        st.dataframe(daily, use_container_width=True, hide_index=True)
        st.download_button("Download daily_inventory.csv", daily.to_csv(index=False).encode("utf-8"), "daily_inventory.csv", "text/csv", key="simulation_download")
    else:
        st.info("Configure this tab and press Run Inventory Simulation.")

with tabs[3]:
    st.header("Strategy Comparison")
    st.caption("Run the three policies on the same fixed final test window and compare modeled cost and service.")
    with st.form("strategy_comparison_form"):
        selected_id, store, dept, item = selector("comparison")
        lead_time = st.number_input("Lead time (days)", min_value=1, max_value=60, value=10, step=1, key="comparison_lead")
        initial_mode = st.selectbox("Initial inventory", ["Auto", "Manual"], key="comparison_initial_mode")
        initial_manual = st.number_input("Manual units", min_value=0.0, value=25.0, step=1.0, disabled=initial_mode != "Manual", key="comparison_initial_manual")
        costs = cost_controls("comparison", selected_frame(selected_id))
        fill_target = st.number_input("Minimum fill rate (0 = off)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="comparison_fill")
        alphas, betas = grid_controls("comparison")
        run_comparison = st.form_submit_button("Run Strategy Comparison", type="primary")
    if run_comparison:
        st.session_state.comparison_result = run_backtest(selected_id, costs, int(lead_time), initial_mode, initial_manual, alphas, betas, fill_target)
    if "comparison_result" in st.session_state:
        frame, opt, test, initial, grid, best, runs, summary = st.session_state.comparison_result
        st.dataframe(summary.style.format({col: "{:.2f}" for col in summary.columns if col != "Strategy"}), use_container_width=True, hide_index=True)
        left, right = st.columns(2)
        with left:
            st.caption("Modeled cost breakdown")
            st.bar_chart(summary.set_index("Strategy")[["ordering_cost", "holding_cost", "stockout_cost"]], height=350)
        with right:
            st.caption("Service and inventory")
            st.bar_chart(summary.set_index("Strategy")[["fill_rate", "average_inventory"]], height=350)
        st.download_button("Download strategy_summary.csv", summary.to_csv(index=False).encode("utf-8"), "strategy_summary.csv", "text/csv", key="comparison_download")
    else:
        st.info("Configure this tab and press Run Strategy Comparison.")

with tabs[4]:
    st.header("Optimization")
    st.caption("Grid search for alpha and beta on optimization d=1830–1913 only. The final test d=1914–1941 is never used to select parameters.")
    with st.form("optimization_form"):
        selected_id, store, dept, item = selector("optimization")
        lead_time = st.number_input("Lead time (days)", min_value=1, max_value=60, value=10, step=1, key="optimization_lead")
        initial_mode = st.selectbox("Initial inventory", ["Auto", "Manual"], key="optimization_initial_mode")
        initial_manual = st.number_input("Manual units", min_value=0.0, value=25.0, step=1.0, disabled=initial_mode != "Manual", key="optimization_initial_manual")
        costs = cost_controls("optimization", selected_frame(selected_id))
        fill_target = st.number_input("Minimum fill rate (0 = off)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="optimization_fill")
        alphas, betas = grid_controls("optimization")
        run_optimization = st.form_submit_button("Run Optimization", type="primary")
    if run_optimization:
        st.session_state.optimization_result = run_backtest(selected_id, costs, int(lead_time), initial_mode, initial_manual, alphas, betas, fill_target)
    if "optimization_result" in st.session_state:
        frame, opt, test, initial, grid, best, runs, summary = st.session_state.optimization_result
        best_cols = st.columns(4)
        best_cols[0].metric("Best alpha", f"{float(best.alpha):.2f}")
        best_cols[1].metric("Best beta", f"{float(best.beta):.2f}")
        best_cols[2].metric("Optimization cost", f"${float(best.total_cost):,.2f}")
        best_cols[3].metric("Optimization fill rate", f"{float(best.fill_rate):.1%}")
        cost_matrix = grid.pivot(index="alpha", columns="beta", values="total_cost")
        st.subheader("Alpha-beta cost matrix")
        st.caption("Rows are alpha values, columns are beta values. Each cell is total modeled cost on the optimization window; lower is better.")
        st.dataframe(cost_matrix.style.format("${:,.2f}"), use_container_width=True)
        st.dataframe(grid.sort_values("total_cost"), use_container_width=True, hide_index=True)
        st.download_button("Download grid_search_results.csv", grid.to_csv(index=False).encode("utf-8"), "grid_search_results.csv", "text/csv", key="optimization_grid_download")
        config = {"series_id": selected_id, "optimization": [OPT_START, OPT_END], "test": [TEST_START, TEST_END], "lead_time": int(lead_time), "initial_inventory": initial, "costs": costs.__dict__, "best_alpha": float(best.alpha), "best_beta": float(best.beta)}
        st.download_button("Download run_config.json", json.dumps(config, indent=2, default=str).encode("utf-8"), "run_config.json", "application/json", key="optimization_config_download")
    else:
        st.info("Configure this tab and press Run Optimization.")

with tabs[5]:
    st.header("Strategy Statistics")
    st.caption("For each item, the recommended strategy is the one with the lowest modeled cost on test d=1914–1941.")
    with st.form("strategy_statistics_form"):
        scope_cols = st.columns(3)
        scope_store = scope_cols[0].selectbox("Store scope", ["All"] + sorted(series["store"].unique()), key="stats_store")
        department_values = sorted(series.loc[series["store"] == scope_store, "department"].unique()) if scope_store != "All" else sorted(series["department"].unique())
        scope_dept = scope_cols[1].selectbox("Department scope", ["All"] + department_values, key="stats_dept")
        lead_time = scope_cols[2].number_input("Lead time (days)", min_value=1, max_value=60, value=10, step=1, key="stats_lead")
        st.caption("Alpha and beta are selected automatically for each item by grid search on the optimization window.")
        run_statistics = st.form_submit_button("Run Strategy Statistics", type="primary")
    if run_statistics:
        candidates = series.copy()
        if scope_store != "All":
            candidates = candidates[candidates["store"] == scope_store]
        if scope_dept != "All":
            candidates = candidates[candidates["department"] == scope_dept]
        progress = st.progress(0.0, text="Starting item-level evaluation")
        st.session_state.statistics_result = batch_strategy_statistics(candidates, int(lead_time), [0.8, 0.9, 1.0, 1.1, 1.2], [0.5, 0.75, 1.0, 1.25, 1.5], progress)
        progress.empty()
    if "statistics_result" in st.session_state:
        stats = st.session_state.statistics_result
        counts = stats.groupby(["store_id", "recommended_strategy"]).size().unstack(fill_value=0).reset_index()
        totals = stats["recommended_strategy"].value_counts().rename_axis("strategy").reset_index(name="item_count")
        st.subheader("Recommended strategy counts")
        k1, k2, k3 = st.columns(3)
        for col, strategy_name in zip([k1, k2, k3], ["Historical baseline", "Forecast-based", "Forecast + alpha-beta"]):
            col.metric(strategy_name, int(totals.loc[totals["strategy"] == strategy_name, "item_count"].sum()))
        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.subheader("All-item benchmark")
        st.caption("Each policy is averaged over the same complete item set; this is the comparable benchmark.")
        st.dataframe(all_item_benchmark(stats), use_container_width=True, hide_index=True)
        st.subheader("Item-level recommendation")
        record_view = stats.rename(columns={
            "recommended_strategy": "Policy",
            "recommended_total_cost": "Total cost",
            "shortage_rate": "Shortage rate",
            "average_inventory": "Average inventory",
        })
        st.dataframe(record_view.sort_values(["store_id", "item_id"]), use_container_width=True, hide_index=True)
        st.download_button("Download strategy_statistics.csv", stats.to_csv(index=False).encode("utf-8"), "strategy_statistics.csv", "text/csv", key="statistics_download")
    else:
        st.info("Choose a scope and press Run Strategy Statistics.")

with tabs[6]:
    st.header("Strategy Visualization")
    st.caption("All-item benchmark: each policy is averaged over the same complete item set.")
    if "statistics_result" in st.session_state:
        stats = st.session_state.statistics_result.copy()
        benchmark = all_item_benchmark(stats)
        policy_summary = benchmark.rename(columns={"mean_total_cost": "total_cost", "mean_shortage_rate": "shortage_rate", "mean_average_inventory": "average_inventory"})
        st.dataframe(policy_summary, use_container_width=True, hide_index=True)
        chart_cols = st.columns(3)
        chart_cols[0].caption("Average total cost by policy")
        chart_cols[0].bar_chart(policy_summary.set_index("Policy")[["total_cost"]], height=320)
        chart_cols[1].caption("Average shortage rate by policy")
        chart_cols[1].bar_chart(policy_summary.set_index("Policy")[["shortage_rate"]], height=320)
        chart_cols[2].caption("Average inventory by policy")
        chart_cols[2].bar_chart(policy_summary.set_index("Policy")[["average_inventory"]], height=320)
    else:
        st.info("Run Strategy Statistics first, then open this tab to view the charts.")

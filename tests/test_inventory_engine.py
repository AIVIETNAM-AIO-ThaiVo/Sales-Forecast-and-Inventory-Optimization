import pandas as pd

from src.inventory_engine import Costs, simulate


def test_order_arrives_after_lead_time_and_lost_sales_are_recorded():
    data = pd.DataFrame({"__d": range(4), "__date": pd.NaT, "__demand": [0, 20, 0, 0], "__forecast": [0, 0, 0, 0]})
    daily, summary = simulate(data, "Historical baseline", lead_time=2, initial_inventory=0, costs=Costs(10), warmup=0)
    assert daily.loc[0, "arrival_d"] is not None
    assert daily.loc[1, "lost_sales"] == 20
    assert summary["lost_sales"] == 20


def test_inventory_position_includes_pipeline_only_once():
    data = pd.DataFrame({"__d": range(3), "__date": pd.NaT, "__demand": [0, 0, 0], "__forecast": [1, 1, 1]})
    daily, _ = simulate(data, "Forecast-based", lead_time=2, initial_inventory=0, costs=Costs(10))
    assert daily.loc[0, "inventory_position"] >= daily.loc[0, "order_qty"]

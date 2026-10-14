from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


@dataclass
class Dataset:
    frame: pd.DataFrame
    id_column: str
    demand_column: str
    trend_column: str | None
    feature_columns: list[str]
    model_paths: dict[tuple[str, str], Path]


def load_dataset(root: str | Path = ".") -> Dataset:
    root = Path(root)
    path = root / "Data" / "df_detrend.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}")
    frame = pd.read_parquet(path)
    id_column = _first_column(frame, ["id_str", "id", "series_id"])
    demand_column = _first_column(frame, ["sold", "sales", "demand", "y"])
    if not id_column or not demand_column:
        raise ValueError("Parquet cần có cột định danh (id_str/id) và demand (sold/sales/demand/y).")
    # Do not copy the 7M-row table here: the extra full-table copy can require
    # another ~800 MB and cause Streamlit startup to fail on normal machines.
    frame["__id"] = frame[id_column] if pd.api.types.is_string_dtype(frame[id_column]) else frame[id_column].astype(str)
    frame["__d"] = frame["d"] if "d" in frame else np.arange(len(frame))
    date_col = _first_column(frame, ["date", "ds", "calendar_date"])
    if date_col:
        frame["__date"] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["__demand"] = pd.to_numeric(frame[demand_column], errors="coerce").fillna(0).clip(lower=0)
    trend_column = "trend" if "trend" in frame.columns else None
    excluded = {id_column, demand_column, "__id", "__d", "__date", "__demand", "y", "sold", "sales", "demand", "trend"}
    feature_columns = [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
    model_paths: dict[tuple[str, str], Path] = {}
    for p in sorted((root / "Models").glob("*.pkl")):
        m = re.search(r"model_([A-Z]+_\d+)_(HOBBIES_[12])", p.stem)
        if m:
            model_paths[(m.group(1), m.group(2))] = p
    return Dataset(frame, id_column, demand_column, trend_column, feature_columns, model_paths)


def parse_group(series_id: str) -> tuple[str, str] | None:
    # M5 ids commonly look like HOBBIES_1_001_CA_1 or CA_1_HOBBIES_1_...
    text = str(series_id)
    m = re.search(r"(CA|TX|WI)_\d+", text)
    d = re.search(r"(HOBBIES_[12])", text)
    if m and d:
        return m.group(0), d.group(1)
    return None


def available_series(ds: Dataset) -> pd.DataFrame:
    rows = []
    for value in ds.frame["__id"].drop_duplicates():
        group = parse_group(value)
        if group:
            store, dept = group
        else:
            store, dept = "Unknown", "Unknown"
        rows.append({"id": value, "store": store, "department": dept})
    return pd.DataFrame(rows).sort_values(["store", "department", "id"]).reset_index(drop=True)


@lru_cache(maxsize=32)
def _load_pickle(path: str) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_model(ds: Dataset, series_id: str) -> tuple[Any | None, str | None]:
    group = parse_group(series_id)
    if not group or group not in ds.model_paths:
        return None, "Không tìm thấy model theo nhóm store/department của SKU này."
    try:
        return _load_pickle(str(ds.model_paths[group])), None
    except Exception as exc:
        return None, f"Không load được model {ds.model_paths[group].name}: {exc}"


def forecast_series(ds: Dataset, series_id: str, frame: pd.DataFrame) -> tuple[pd.Series, str]:
    model, warning = load_model(ds, series_id)
    if model is not None:
        names = list(getattr(model, "feature_name_", []) or [])
        if not names and callable(getattr(model, "feature_name", None)):
            names = list(model.feature_name())
        # The artifact expects `base`, while the stored feature table keeps the
        # detrended target as y and trend separately. Use only lagged/expanding
        # history to construct a leakage-safe approximation for this MVP.
        model_frame = frame.copy()
        if "base" not in model_frame.columns:
            if "rolling_sold_mean_7" in model_frame.columns:
                model_frame["base"] = model_frame["rolling_sold_mean_7"]
            elif "expanding_sold_mean" in model_frame.columns:
                model_frame["base"] = model_frame["expanding_sold_mean"]
            else:
                model_frame["base"] = 0.0
        if names and all(c in model_frame.columns for c in names):
            x = model_frame[names].apply(pd.to_numeric, errors="coerce").fillna(0)
            try:
                pred = np.asarray(model.predict(x), dtype=float)
                if ds.trend_column and ds.trend_column in frame:
                    pred = pred + pd.to_numeric(frame[ds.trend_column], errors="coerce").fillna(0).to_numpy()
                return pd.Series(np.maximum(pred, 0), index=frame.index), "LightGBM artifact"
            except Exception as exc:
                warning = f"Model có nhưng prediction lỗi: {exc}"
        else:
            warning = "Model feature_name_ không khớp parquet; dùng fallback seasonal-naive."
    # Transparent fallback keeps the app usable when a model runtime/schema is unavailable.
    base = frame["__demand"].astype(float)
    pred = base.shift(7).fillna(base.rolling(7, min_periods=1).mean()).fillna(0)
    return pred.clip(lower=0), warning or "Dùng fallback seasonal-naive."


@dataclass
class Costs:
    reference_price: float
    cost_ratio: float = 0.76
    holding_rate: float = 0.25
    ordering_cost: float = 5.0
    stockout_penalty: float | None = None

    @property
    def unit_cost(self) -> float:
        return self.cost_ratio * self.reference_price

    @property
    def daily_holding(self) -> float:
        return self.holding_rate * self.unit_cost / 365.0

    @property
    def penalty(self) -> float:
        return self.stockout_penalty if self.stockout_penalty is not None else max(0.0, self.reference_price - self.unit_cost)


def policy_parameters(history: pd.Series, forecast: pd.Series, lead_time: int, costs: Costs, z: float = 1.645) -> dict[str, float]:
    mean = float(history.mean()) if len(history) else 0.0
    std = float(history.std(ddof=1)) if len(history) > 1 else 0.0
    ss1 = max(0.0, z * std * math.sqrt(lead_time))
    rop1 = math.ceil(mean * lead_time + ss1)
    annual = max(mean * 365.0, 0.0)
    qa = math.sqrt(2 * costs.ordering_cost * annual / max(costs.holding_rate * costs.unit_cost, 1e-9))
    q1 = max(1, round(qa))
    horizon = min(lead_time, len(forecast))
    f10 = float(forecast.iloc[:horizon].sum()) if horizon else 0.0
    error = history.iloc[-min(len(history), len(forecast)):].sum() - forecast.iloc[:min(len(history), len(forecast))].sum() if len(history) else 0.0
    ss2 = max(0.0, float(error))
    rop2 = math.ceil(max(0.0, f10 + ss2))
    annual_f = max(f10 / max(horizon, 1) * 365.0, 0.0)
    q2 = max(1, round(math.sqrt(2 * costs.ordering_cost * annual_f / max(costs.holding_rate * costs.unit_cost, 1e-9))))
    return {"rop1": rop1, "q1": q1, "rop2": rop2, "q2": q2, "ss1": ss1, "ss2": ss2}


def simulate(data: pd.DataFrame, strategy: str, lead_time: int, initial_inventory: float, costs: Costs, alpha: float = 1.0, beta: float = 1.0, warmup: int = 0) -> tuple[pd.DataFrame, dict[str, float]]:
    data = data.sort_values("__d").reset_index(drop=True).copy()
    history = data["__demand"].iloc[:warmup]
    on_hand = float(initial_inventory)
    pipeline: dict[int, float] = {}
    rows = []
    for i, row in data.iloc[warmup:].iterrows():
        arrivals = pipeline.pop(int(row["__d"]), 0.0)
        on_hand += arrivals
        demand = float(row["__demand"])
        if "__lead_forecast" in data.columns and pd.notna(row.get("__lead_forecast")):
            # The policy consumes the forecast total for exactly the current
            # lead-time window. Padding keeps the annualization denominator at L.
            forecast_window = pd.Series([float(row["__lead_forecast"])] + [0.0] * max(lead_time - 1, 0))
        else:
            forecast_window = data["__forecast"].iloc[i + 1 : i + lead_time + 1]
        pars = policy_parameters(history, forecast_window, lead_time, costs)
        if strategy == "Historical baseline":
            rop, qty = pars["rop1"], pars["q1"]
        else:
            rop, qty = pars["rop2"], pars["q2"]
            if strategy == "Forecast + alpha-beta":
                rop, qty = math.ceil(alpha * rop), max(1, round(beta * qty))
        fulfilled = min(demand, on_hand)
        lost = max(0.0, demand - fulfilled)
        on_hand -= fulfilled
        outstanding = float(sum(pipeline.values()))
        order = 0.0
        arrival_day = None
        if on_hand + outstanding <= rop:
            order = float(qty)
            arrival_day = int(row["__d"]) + lead_time
            pipeline[arrival_day] = pipeline.get(arrival_day, 0.0) + order
        ip = on_hand + float(sum(pipeline.values()))
        rows.append({"d": int(row["__d"]), "date": row["__date"], "demand": demand, "forecast": float(row["__forecast"]), "forecast_lead_total": float(row.get("__lead_forecast", forecast_window.sum())), "arrivals": arrivals, "on_hand": on_hand, "inventory_position": ip, "rop": rop, "order_qty": order, "arrival_d": arrival_day, "fulfilled": fulfilled, "lost_sales": lost, "ordering_cost": costs.ordering_cost if order else 0.0, "holding_cost": costs.daily_holding * on_hand, "stockout_cost": costs.penalty * lost})
        history = pd.concat([history, pd.Series([demand])], ignore_index=True).tail(365)
    out = pd.DataFrame(rows)
    summary = {"total_cost": float(out[["ordering_cost", "holding_cost", "stockout_cost"]].sum().sum()), "ordering_cost": float(out["ordering_cost"].sum()), "holding_cost": float(out["holding_cost"].sum()), "stockout_cost": float(out["stockout_cost"].sum()), "fill_rate": float(1 - out["lost_sales"].sum() / max(out["demand"].sum(), 1e-9)), "orders": int((out["order_qty"] > 0).sum()), "average_inventory": float(out["on_hand"].mean()), "lost_sales": float(out["lost_sales"].sum())}
    return out, summary


def grid_search(data: pd.DataFrame, lead_time: int, initial_inventory: float, costs: Costs, alphas: list[float], betas: list[float], fill_target: float | None = None) -> pd.DataFrame:
    records = []
    for alpha in alphas:
        for beta in betas:
            _, s = simulate(data, "Forecast + alpha-beta", lead_time, initial_inventory, costs, alpha, beta)
            records.append({"alpha": alpha, "beta": beta, **s, "feasible": fill_target is None or s["fill_rate"] >= fill_target})
    return pd.DataFrame(records)

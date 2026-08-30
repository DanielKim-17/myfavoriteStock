import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.set_page_config(page_title="My Favorite Stock", layout="wide")


SHEET_NAME = os.getenv("MYFAVORITE_SHEET_NAME", "myfavorite")
CREDENTIAL_ENV_KEYS = (
    "GOOGLE_SERVICE_ACCOUNT_JSON",
    "GOOGLE_SHEET_CREDENTIALS",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def normalize_ticker(raw: str) -> str:
    """Yahoo Finance 티커를 정규화한다."""
    if raw is None:
        return ""
    ticker = str(raw).strip()
    if not ticker:
        return ""
    ticker = ticker.replace(" ", "")
    if "." in ticker:
        return ticker.upper()
    if ticker.isdigit():
        return f"{ticker}.KS"
    return ticker.upper()


def _read_json_from_string(value: str) -> Optional[dict]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


def get_service_account_info() -> Optional[dict]:
    for key in CREDENTIAL_ENV_KEYS:
        value = os.getenv(key)
        if value:
            if value.startswith("{"):
                info = _read_json_from_string(value)
                if info:
                    return info
            path = Path(value)
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    if isinstance(info, dict):
                        return info
                except Exception:
                    pass

    default_candidates = [
        Path(__file__).resolve().parent / "google-service-account.json",
        Path.cwd() / "google-service-account.json",
        Path.home() / "google-service-account.json",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    info = json.load(f)
                if isinstance(info, dict):
                    return info
            except Exception:
                pass

    try:
        secrets = st.secrets
        direct_keys = ("google_service_account", "gcp_service_account", "service_account", "google_service_account_json")
        for key in direct_keys:
            if key in dict(secrets):
                value = secrets[key]
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    info = _read_json_from_string(value)
                    if info:
                        return info

        if "connections" in dict(secrets):
            conn = secrets["connections"]
            for key in ("gsheets", "google_sheets", "service_account"):
                if key in dict(conn):
                    value = conn[key]
                    if isinstance(value, dict):
                        return value

        if "gsheets" in dict(secrets):
            value = secrets["gsheets"]
            if isinstance(value, dict):
                return value
    except Exception:
        pass

    return None


@st.cache_data(ttl=600)
def load_favorite_sheet() -> pd.DataFrame:
    """Google Sheet의 myfavorite 시트를 읽어 카테고리와 종목 정보를 반환한다."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    service_account_info = get_service_account_info()
    if not service_account_info:
        raise RuntimeError(
            "Google Service Account 정보가 없습니다. "
            ".streamlit/secrets.toml, GOOGLE_SERVICE_ACCOUNT_JSON, "
            "또는 google-service-account.json 경로를 설정해 주세요."
        )

    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    client = gspread.authorize(creds)

    workbook = client.open(SHEET_NAME)
    worksheet = workbook.sheet1
    rows = worksheet.get_all_records()

    if not rows:
        return pd.DataFrame(columns=["Category", "Ticker", "Ticker Name", "Name"])

    df = pd.DataFrame(rows)
    rename_map = {}
    for col in list(df.columns):
        key = str(col).strip().lower().replace(" ", "")
        if key in {"category", "ticker", "name", "tickername"}:
            rename_map[col] = {"category": "Category", "ticker": "Ticker", "name": "Ticker Name", "tickername": "Ticker Name"}.get(key, col)
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Category" not in df.columns:
        raise ValueError("Google Sheet에 'Category' 컬럼이 없습니다.")
    if "Ticker" not in df.columns:
        raise ValueError("Google Sheet에 'Ticker' 컬럼이 없습니다.")

    if "Ticker Name" not in df.columns:
        df["Ticker Name"] = ""

    df["Category"] = df["Category"].fillna("").astype(str).str.strip()
    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip()
    df["Ticker Name"] = df["Ticker Name"].fillna("").astype(str).str.strip()
    df = df[df["Ticker"] != ""].copy()
    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    return df.reset_index(drop=True)


@st.cache_data(ttl=300)
def fetch_daily_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """최근 1년치 주가 데이터를 yfinance에서 가져온다."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval="1d", auto_adjust=False, actions=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if not isinstance(hist, pd.DataFrame):
        return pd.DataFrame()

    hist = hist.copy()
    if isinstance(hist.index, pd.DatetimeIndex):
        hist = hist.reset_index()
    elif hist.index.name is not None:
        hist = hist.rename_axis("Date").reset_index()

    hist.columns = [str(c).strip() for c in hist.columns]

    if "Datetime" in hist.columns and "Date" not in hist.columns:
        hist = hist.rename(columns={"Datetime": "Date"})
    if "Date" not in hist.columns and hist.index.name is not None:
        hist = hist.rename_axis("Date").reset_index()
    if "Date" not in hist.columns:
        return pd.DataFrame()

    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    hist = hist.dropna(subset=["Date"]).copy()
    if hist.empty:
        return pd.DataFrame()

    hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
    hist = hist.sort_values("Date").reset_index(drop=True)
    hist["Ticker"] = ticker
    return hist


@st.cache_data(ttl=300)
def build_stock_metrics(ticker: str) -> pd.DataFrame:
    """티커별 기술지표를 계산한다."""
    hist = fetch_daily_history(ticker, period="1y")
    if hist.empty:
        return pd.DataFrame()

    hist = hist.copy()
    hist["Open"] = pd.to_numeric(hist.get("Open", pd.Series(index=hist.index, dtype=float)), errors="coerce")
    hist["High"] = pd.to_numeric(hist.get("High", pd.Series(index=hist.index, dtype=float)), errors="coerce")
    hist["Low"] = pd.to_numeric(hist.get("Low", pd.Series(index=hist.index, dtype=float)), errors="coerce")
    hist["Close"] = pd.to_numeric(hist.get("Close", pd.Series(index=hist.index, dtype=float)), errors="coerce")
    hist["Volume"] = pd.to_numeric(hist.get("Volume", pd.Series(index=hist.index, dtype=float)), errors="coerce")

    hist = hist.dropna(subset=["Close", "High", "Low", "Volume"]).reset_index(drop=True)
    if hist.empty:
        return pd.DataFrame()

    hist["PrevClose"] = hist["Close"].shift(1)
    hist["TR1"] = hist["High"] - hist["Low"]
    hist["TR2"] = (hist["High"] - hist["PrevClose"]).abs()
    hist["TR3"] = (hist["Low"] - hist["PrevClose"]).abs()

    n_value = hist.apply(
        lambda r: max(
            float(r["High"] - r["Low"]),
            abs(float(r["High"] - r["PrevClose"])),
            abs(float(r["Low"] - r["PrevClose"])),
        ),
        axis=1,
    )
    hist["Nvalue"] = n_value
    hist["NvalueAvg20"] = hist["Nvalue"].rolling(window=20, min_periods=20).mean()
    hist["NvalueAbs"] = hist["NvalueAvg20"].abs()
    hist["NrateAbs"] = (hist["NvalueAvg20"] / hist["Close"]).abs().replace([np.inf, -np.inf], np.nan)
    hist["Min10"] = hist["Low"].rolling(window=10, min_periods=10).min()
    hist["Min20"] = hist["Low"].rolling(window=20, min_periods=20).min()
    hist["Moving28"] = hist["Close"].rolling(window=28, min_periods=28).mean()
    hist["stoploss"] = hist[["Min10", "Moving28"]].min(axis=1)

    delta = hist["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    hist["RSI"] = 100 - (100 / (1 + rs))
    hist["RSI"] = hist["RSI"].replace([np.inf, -np.inf], 50)

    hist["MB"] = hist["Close"].rolling(window=20, min_periods=20).mean()
    std_20 = hist["Close"].rolling(window=20, min_periods=20).std(ddof=0)
    hist["UB"] = hist["MB"] + 2 * std_20
    hist["LB"] = hist["MB"] - 2 * std_20

    diff = hist["Close"].diff()
    volume = hist["Volume"].fillna(0)
    direction = np.where(diff > 0, volume, np.where(diff < 0, -volume, 0))
    hist["OBV"] = pd.Series(direction, index=hist.index).cumsum()

    tp = (hist["High"] + hist["Low"] + hist["Close"]) / 3
    mf = tp * volume
    tp_diff = tp.diff()
    pmf = np.where(tp_diff > 0, mf, 0)
    nmf = np.where(tp_diff < 0, mf, 0)
    pmf_sum = pd.Series(pmf, index=hist.index).rolling(window=14, min_periods=14).sum()
    nmf_sum = pd.Series(nmf, index=hist.index).rolling(window=14, min_periods=14).sum()
    mfi_den = pmf_sum + nmf_sum
    hist["MFI"] = np.where(mfi_den == 0, 50, 100 * pmf_sum / mfi_den)

    return hist


def get_category_table() -> pd.DataFrame:
    favorites = load_favorite_sheet()
    categories = sorted(favorites["Category"].dropna().unique().tolist())
    st.sidebar.subheader("Category")
    selected_category = st.sidebar.selectbox("Category 선택", categories, index=0 if categories else None)

    selection_df = favorites[favorites["Category"] == selected_category].copy()
    ticker_names = {}
    for _, row in selection_df.iterrows():
        ticker_names[row["Ticker"]] = row.get("Ticker Name", "") or ""

    rows = []
    for ticker in selection_df["Ticker"].tolist():
        hist = build_stock_metrics(ticker)
        if hist.empty:
            continue
        last = hist.iloc[-1]
        row = {
            "Ticker": ticker,
            "Ticker Name": ticker_names.get(ticker, ""),
            "현재가": last["Close"],
            "Min10": last["Min10"],
            "Min20": last["Min20"],
            "Moving28": last["Moving28"],
            "stoploss": last["stoploss"],
            "NvalueAbs": last["NvalueAbs"],
            "NrateAbs": last["NrateAbs"],
            "TR1": last["TR1"],
            "TR2": last["TR2"],
            "TR3": last["TR3"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning(f"'{selected_category}' 카테고리에 해당하는 티커가 없습니다.")
        st.stop()

    df = df.sort_values("현재가", ascending=False, na_position="last").reset_index(drop=True)
    return df, selected_category, selection_df


def build_detail_chart(df: pd.DataFrame, ticker: str) -> None:
    hist = build_stock_metrics(ticker)
    if hist.empty:
        st.warning(f"{ticker}의 데이터를 가져오지 못했습니다.")
        return

    hist = hist.ffill().bfill()
    hist = hist.tail(252).copy()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.20, 0.15, 0.15],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    x_dates = pd.to_datetime(hist["Date"])

    fig.add_trace(
        go.Candlestick(
            x=x_dates,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="Candles",
            increasing_line_color="red",
            decreasing_line_color="blue",
            increasing_fillcolor="red",
            decreasing_fillcolor="blue",
            opacity=0.9,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=x_dates, y=hist["Moving28"], mode="lines", name="Moving28", line=dict(color="darkorange", width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["MB"], mode="lines", name="MB", line=dict(color="green", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["UB"], mode="lines", name="UB", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["LB"], mode="lines", name="LB", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)

    fig.add_trace(go.Bar(x=x_dates, y=hist["Volume"], name="Volume", marker_color="lightgray"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["OBV"], mode="lines", name="OBV", line=dict(color="purple", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["RSI"], mode="lines", name="RSI", line=dict(color="firebrick", width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["MFI"], mode="lines", name="MFI", line=dict(color="teal", width=2)), row=4, col=1)

    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        rangeslider_visible=False,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        rangebreaks=[{"bounds": ["sat", "mon"]}],
        row=2,
        col=1,
    )
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        row=3,
        col=1,
    )
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        row=4,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker} 최근 1년 차트 (Candles / Volume / OBV / RSI / MFI)",
        template="plotly_white",
        height=950,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="OBV", row=3, col=1)
    fig.update_yaxes(title_text="Indicator", row=4, col=1)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    try:
        favorites = load_favorite_sheet()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.title("My Favorite Stock")
    st.caption("Google Sheet의 myfavorite 데이터를 읽어 yfinance로 실시간 지표를 계산합니다.")

    if favorites.empty:
        st.warning("myfavorite 시트에 데이터가 없습니다.")
        st.stop()

    categories = sorted(favorites["Category"].dropna().unique().tolist())
    selected_category = st.selectbox("Category 선택", categories)
    selected_df = favorites[favorites["Category"] == selected_category].copy()

    rows = []
    for ticker in selected_df["Ticker"].tolist():
        hist = build_stock_metrics(ticker)
        if hist.empty:
            continue
        last = hist.iloc[-1]
        name = selected_df.loc[selected_df["Ticker"] == ticker, "Ticker Name"].iloc[0] if not selected_df.loc[selected_df["Ticker"] == ticker].empty else ""
        rows.append(
            {
                "Ticker": ticker,
                "현재가": last["Close"],
                "Min10": last["Min10"],
                "Min20": last["Min20"],
                "Moving28": last["Moving28"],
                "stoploss": last["stoploss"],
                "NvalueAbs": last["NvalueAbs"],
                "NrateAbs": last["NrateAbs"],
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        st.warning(f"'{selected_category}' 카테고리에서 유효한 종목을 찾지 못했습니다.")
        st.stop()

    summary = summary.sort_values("현재가", ascending=False, na_position="last").reset_index(drop=True)
    summary_display = summary.copy()
    numeric_cols = ["현재가", "Min10", "Min20", "Moving28", "stoploss", "NvalueAbs", "NrateAbs"]
    for col in numeric_cols:
        if col in summary_display.columns:
            summary_display[col] = summary_display[col].map(lambda x: round(float(x), 2) if pd.notna(x) else x)

    st.subheader(f"'{selected_category}' 종목 스크리닝")

    data_for_table = summary_display.copy()
    selected = st.dataframe(
        data_for_table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if not selected.selection.rows:
        st.info("표에서 종목을 선택하면 최근 1년 추세 차트를 표시합니다.")
        st.stop()

    selected_row_idx = selected.selection.rows[0]
    selected_ticker = data_for_table.loc[selected_row_idx, "Ticker"]
    st.subheader(f"선택 종목: {selected_ticker}")
    build_detail_chart(summary, selected_ticker)


if __name__ == "__main__":
    main()

import streamlit as st
import yfinance as yf
import json
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="估值扫描",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SECTOR_CN = {
    "Technology": "科技", "Healthcare": "医疗健康",
    "Financial Services": "金融", "Consumer Cyclical": "消费（周期）",
    "Consumer Defensive": "消费（必需）", "Industrials": "工业",
    "Energy": "能源", "Utilities": "公用事业",
    "Real Estate": "房地产", "Communication Services": "通信",
    "Basic Materials": "原材料",
}

SECTOR_FAIR_PE = {
    "Technology": 28,
    "Healthcare": 22,
    "Financial Services": 14,
    "Consumer Cyclical": 22,
    "Consumer Defensive": 20,
    "Industrials": 20,
    "Energy": 12,
    "Utilities": 16,
    "Real Estate": 30,
    "Communication Services": 22,
    "Basic Materials": 14,
}

# ── Supabase ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_supabase():
    from supabase import create_client
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def load_tickers(profile: str) -> list:
    defaults = ["AAPL", "NVDA", "MSFT", "SPY", "QQQ"]
    try:
        sb = get_supabase()
        res = sb.table("profiles").select("tickers").eq("name", profile).execute()
        if res.data:
            return res.data[0]["tickers"]
        sb.table("profiles").insert({"name": profile, "tickers": defaults}).execute()
        load_all_profiles.clear()
        return defaults
    except Exception:
        return defaults

def save_tickers(profile: str, tickers: list):
    try:
        get_supabase().table("profiles").upsert({"name": profile, "tickers": tickers}).execute()
    except Exception as e:
        st.error(f"保存失败：{e}")

@st.cache_data(ttl=60)
def load_all_profiles() -> list:
    try:
        res = get_supabase().table("profiles").select("name").execute()
        return [r["name"] for r in res.data]
    except Exception:
        return []

def delete_profile(name: str):
    try:
        get_supabase().table("profiles").delete().eq("name", name).execute()
        load_all_profiles.clear()
    except Exception as e:
        st.error(f"删除失败：{e}")

def rename_profile(old_name: str, new_name: str):
    try:
        sb = get_supabase()
        res = sb.table("profiles").select("tickers").eq("name", old_name).execute()
        tickers = res.data[0]["tickers"] if res.data else []
        sb.table("profiles").insert({"name": new_name, "tickers": tickers}).execute()
        sb.table("profiles").delete().eq("name", old_name).execute()
        load_all_profiles.clear()
    except Exception as e:
        st.error(f"重命名失败：{e}")

# ── data fetching ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_yf_session():
    from curl_cffi import requests as cf_requests
    return cf_requests.Session(impersonate="chrome")

@st.cache_data(ttl=300)
def fetch_info(symbol: str):
    try:
        return yf.Ticker(symbol, session=get_yf_session()).info
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def fetch_history(symbol: str, period: str = "2y"):
    try:
        return yf.Ticker(symbol, session=get_yf_session()).history(period=period)
    except Exception:
        return pd.DataFrame()

# ── valuation engine ───────────────────────────────────────────────────────────

def score_stock(info: dict):
    """
    Returns (verdict, emoji, score, signals).
    score < 0 = cheap side, score > 0 = expensive side.
    """
    score = 0.0
    signals = []
    is_etf = info.get("quoteType") == "ETF"

    if not is_etf:
        fair_pe = SECTOR_FAIR_PE.get(info.get("sector", ""), 20)

        trailing_pe = info.get("trailingPE")
        forward_pe  = info.get("forwardPE")
        peg         = info.get("trailingPegRatio")
        ev_ebitda   = info.get("enterpriseToEbitda")

        # Trailing P/E vs sector benchmark
        if trailing_pe and 0 < trailing_pe < 500:
            ratio = trailing_pe / fair_pe
            if ratio < 0.75:
                score -= 1.5
                signals.append(f"P/E {trailing_pe:.1f} — 显著低于行业均值 {fair_pe}（偏便宜）")
            elif ratio < 0.9:
                score -= 0.7
                signals.append(f"P/E {trailing_pe:.1f} — 略低于行业均值 {fair_pe}")
            elif ratio > 1.5:
                score += 1.5
                signals.append(f"P/E {trailing_pe:.1f} — 显著高于行业均值 {fair_pe}（偏贵）")
            elif ratio > 1.2:
                score += 0.7
                signals.append(f"P/E {trailing_pe:.1f} — 略高于行业均值 {fair_pe}")

        # Forward vs trailing P/E direction
        if forward_pe and trailing_pe and 0 < forward_pe < 300 and trailing_pe > 0:
            trend = forward_pe / trailing_pe
            if trend < 0.85:
                score -= 0.5
                signals.append(f"盈利加速：Forward P/E {forward_pe:.1f} < Trailing {trailing_pe:.1f}")
            elif trend > 1.15:
                score += 0.5
                signals.append(f"盈利放缓：Forward P/E {forward_pe:.1f} > Trailing {trailing_pe:.1f}")

        # PEG
        if peg and 0 < peg < 20:
            if peg < 1.0:
                score -= 1.0
                signals.append(f"PEG {peg:.2f} < 1 — 成长调整后偏便宜")
            elif peg > 2.5:
                score += 1.0
                signals.append(f"PEG {peg:.2f} > 2.5 — 成长调整后偏贵")
            else:
                signals.append(f"PEG {peg:.2f} — 成长调整后合理")

        # EV/EBITDA
        if ev_ebitda and 0 < ev_ebitda < 150:
            if ev_ebitda < 8:
                score -= 1.0
                signals.append(f"EV/EBITDA {ev_ebitda:.1f}x — 较低")
            elif ev_ebitda > 35:
                score += 1.0
                signals.append(f"EV/EBITDA {ev_ebitda:.1f}x — 较高")
            else:
                signals.append(f"EV/EBITDA {ev_ebitda:.1f}x")

    # 52-week range position (stocks + ETFs)
    high  = info.get("fiftyTwoWeekHigh")
    low   = info.get("fiftyTwoWeekLow")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    pct52 = None
    if high and low and price and high != low:
        pct52 = (price - low) / (high - low)
        if pct52 < 0.2:
            score -= 1.0
            signals.append(f"价格在52周低位区（{pct52*100:.0f}%）")
        elif pct52 < 0.4:
            score -= 0.3
            signals.append(f"价格在52周中低位（{pct52*100:.0f}%）")
        elif pct52 > 0.85:
            score += 1.0
            signals.append(f"价格在52周高位区（{pct52*100:.0f}%）")
        elif pct52 > 0.7:
            score += 0.3
            signals.append(f"价格在52周中高位（{pct52*100:.0f}%）")

    # Analyst consensus (1=Strong Buy … 5=Strong Sell)
    rec = info.get("recommendationMean")
    if rec:
        if rec <= 1.8:
            score -= 0.5
            signals.append(f"分析师共识：强力买入（{rec:.1f}/5）")
        elif rec >= 3.5:
            score += 0.5
            signals.append(f"分析师共识：偏卖出（{rec:.1f}/5）")
        else:
            signals.append(f"分析师共识：持有/买入（{rec:.1f}/5）")

    # Verdict
    if score <= -2.0:
        return "偏便宜", "🟢", score, pct52, signals
    elif score <= -0.8:
        return "略便宜", "🟡", score, pct52, signals
    elif score < 0.8:
        return "合理区间", "⚪", score, pct52, signals
    elif score < 2.0:
        return "略贵", "🟠", score, pct52, signals
    else:
        return "偏贵", "🔴", score, pct52, signals

# ── helpers ────────────────────────────────────────────────────────────────────

def fmt(val, digits=1, suffix=""):
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    return f"{val:.{digits}f}{suffix}"

def _build_indicators(h):
    h = h.copy()
    h["MA50"]  = h["Close"].ewm(span=50,  adjust=False).mean()
    h["MA200"] = h["Close"].ewm(span=200, adjust=False).mean()
    bb_mid = h["Close"].rolling(20).mean()
    bb_std = h["Close"].rolling(20).std()
    h["BB_upper"] = bb_mid + 2 * bb_std
    h["BB_mid"]   = bb_mid
    h["BB_lower"] = bb_mid - 2 * bb_std
    return h

def _calc_vp(h, current_price):
    n_bins = 60
    p_min, p_max = h["Low"].min(), h["High"].max()
    bins = np.linspace(p_min, p_max, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    vp = np.zeros(n_bins)
    for _, row in h.iterrows():
        mask = (bins[1:] >= row["Low"]) & (bins[:-1] <= row["High"])
        n = mask.sum()
        if n > 0:
            vp[mask] += row["Volume"] / n
    poc = centers[vp.argmax()]
    vp_norm = vp / vp.max()
    colors = ["rgba(100,180,255,0.5)" if c <= current_price else "rgba(255,120,120,0.5)" for c in centers]
    return centers, vp, vp_norm, poc, colors

def _add_chart_traces(fig, h, centers, vp, vp_norm, poc, bar_colors, visible):
    v = visible
    fig.add_trace(go.Scatter(
        x=list(h.index) + list(h.index[::-1]),
        y=list(h["BB_upper"]) + list(h["BB_lower"][::-1]),
        fill="toself", fillcolor="rgba(150,150,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False, visible=v,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.index, y=h["BB_upper"], name="BB上轨", line=dict(color="rgba(150,150,255,0.5)", width=1), visible=v), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.index, y=h["BB_mid"],   name="BB中轨", line=dict(color="rgba(150,150,255,0.7)", width=1, dash="dot"), visible=v), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.index, y=h["BB_lower"], name="BB下轨", line=dict(color="rgba(150,150,255,0.5)", width=1), visible=v), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.index, y=h["Close"],    name="收盘价", line=dict(color="#4da3ff", width=2), visible=v), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.index, y=h["MA50"],     name="EMA50",  line=dict(color="#f0a128", width=1.5, dash="dot"), visible=v), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.index, y=h["MA200"],    name="EMA200", line=dict(color="#e46f95", width=1.5, dash="dash"), visible=v), row=1, col=1)
    fig.add_trace(go.Bar(x=vp_norm, y=centers, orientation="h", marker_color=bar_colors,
        marker_line_width=0, showlegend=False, visible=v,
        hovertemplate="价格 $%{y:.2f}<br>成交量 %{customdata:,.0f}<extra></extra>", customdata=vp,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[h.index[0], h.index[-1]], y=[poc, poc],
        mode="lines", line=dict(color="#FFD700", width=1.2, dash="dash"),
        name=f"POC ${poc:.2f}", showlegend=False, visible=v,
    ), row=1, col=1)

def price_chart(symbol: str, info: dict):
    raw = fetch_history(symbol, "5y")
    if raw.empty:
        st.caption("无法获取历史数据")
        return
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or raw["Close"].iloc[-1]

    d = _build_indicators(raw).iloc[-252:]
    dc, dv, dvn, d_poc, d_clr = _calc_vp(d, current_price)

    wraw = raw.resample("W").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    w = _build_indicators(wraw).iloc[-156:]
    wc, wv, wvn, w_poc, w_clr = _calc_vp(w, current_price)

    N = 9
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], horizontal_spacing=0.0)
    _add_chart_traces(fig, d, dc, dv, dvn, d_poc, d_clr, True)
    _add_chart_traces(fig, w, wc, wv, wvn, w_poc, w_clr, False)

    fig.update_layout(
        height=340, margin=dict(t=10, b=10, l=0, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc", size=11), legend=dict(orientation="h", y=1.08), bargap=0,
        updatemenus=[dict(
            type="buttons", direction="left",
            buttons=[
                dict(label="日线 1年", method="update", args=[{"visible": [True]*N + [False]*N}]),
                dict(label="周线 3年", method="update", args=[{"visible": [False]*N + [True]*N}]),
            ],
            bgcolor="#2a2a2a", bordercolor="#555", font=dict(color="#ccc", size=11),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        )],
    )
    fig.update_xaxes(gridcolor="#2a2a2a", row=1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
    fig.update_yaxes(gridcolor="#2a2a2a", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

# ── profile gate ──────────────────────────────────────────────────────────────

profile = st.query_params.get("profile", "").strip()

if not profile:
    st.markdown("## 📊 估值扫描")

    existing = load_all_profiles()
    if existing:
        st.markdown("**选择用户**")
        cols = st.columns([1] * len(existing) + [max(6 - len(existing), 1)])
        for i, p in enumerate(existing):
            if cols[i].button(p, use_container_width=True, key=f"land_{p}"):
                st.query_params["profile"] = p
                st.rerun()
        st.divider()

    st.markdown("**新建用户**")
    c1, c2 = st.columns([3, 1])
    name_input = c1.text_input("昵称", placeholder="小明 / Alex ...", label_visibility="collapsed")
    if c2.button("进入", use_container_width=True) and name_input.strip():
        st.query_params["profile"] = name_input.strip()
        st.rerun()
    st.stop()

# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 估值扫描")
    st.caption(f"当前用户：**{profile}**")
    if st.button("切换用户", use_container_width=True):
        st.query_params.clear()
        st.rerun()
    st.divider()

    tickers = load_tickers(profile)

    st.subheader("添加")
    col1, col2 = st.columns([3, 1])
    new = col1.text_input("代码", label_visibility="collapsed", placeholder="TSLA").upper().strip()
    if col2.button("＋", use_container_width=True) and new:
        if new not in tickers:
            tickers.append(new)
            save_tickers(profile, tickers)
            st.cache_data.clear()
        st.rerun()

    st.subheader("关注列表")
    from collections import defaultdict
    groups = defaultdict(list)
    for sym in tickers:
        info = fetch_info(sym)
        if not info:
            groups["其他"].append(sym)
        elif info.get("quoteType") == "ETF":
            groups["ETF"].append(sym)
        else:
            en = info.get("sector", "其他")
            groups[SECTOR_CN.get(en, en)].append(sym)

    for sector in sorted(groups):
        st.caption(sector)
        for sym in groups[sector]:
            c1, c2 = st.columns([4, 1])
            c1.write(f"**{sym}**")
            if c2.button("✕", key=f"del_{sym}"):
                tickers.remove(sym)
                save_tickers(profile, tickers)
                st.rerun()

    st.divider()

    st.subheader("用户列表")
    for p in load_all_profiles():
        c1, c2, c3 = st.columns([3, 1, 1])
        if p == profile:
            c1.markdown(f"**→ {p}**")
        else:
            if c1.button(p, key=f"switch_{p}", use_container_width=True):
                st.query_params["profile"] = p
                st.rerun()
        if c2.button("✏️", key=f"edit_{p}", help="重命名"):
            st.session_state["editing_profile"] = p
        if c3.button("🗑", key=f"del_{p}", help="删除"):
            delete_profile(p)
            if p == profile:
                st.query_params.clear()
            st.rerun()

    editing = st.session_state.get("editing_profile")
    if editing:
        st.caption(f"重命名「{editing}」")
        new_name = st.text_input("新名字", value=editing, key="rename_input", label_visibility="collapsed")
        ca, cb = st.columns(2)
        if ca.button("保存", use_container_width=True):
            if new_name.strip() and new_name.strip() != editing:
                rename_profile(editing, new_name.strip())
                if editing == profile:
                    st.query_params["profile"] = new_name.strip()
            del st.session_state["editing_profile"]
            st.rerun()
        if cb.button("取消", use_container_width=True):
            del st.session_state["editing_profile"]
            st.rerun()
    st.divider()

    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("数据来源：Yahoo Finance\n5分钟缓存 · 历史图1小时缓存")

# ── main ───────────────────────────────────────────────────────────────────────

st.title("股票 / ETF 估值快照")
_ny = datetime.now(ZoneInfo("America/New_York"))
st.caption(f"更新于 {_ny.strftime('%Y-%m-%d %H:%M')} (NY时间)")

if not tickers:
    st.info("在左侧添加股票代码开始扫描")
    st.stop()

# Fetch and score all tickers
rows = []
with st.spinner("拉取数据中…"):
    for sym in tickers:
        info = fetch_info(sym)
        if not info or not info.get("regularMarketPrice") and not info.get("currentPrice"):
            st.warning(f"{sym}：无法获取数据，跳过")
            continue

        verdict, emoji, score, pct52, signals = score_stock(info)
        price  = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        is_etf = info.get("quoteType") == "ETF"

        rows.append(dict(
            sym=sym,
            name=(info.get("shortName") or info.get("longName") or sym)[:24],
            price=price,
            sector="— ETF —" if is_etf else info.get("sector", "—"),
            trailing_pe=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            peg=info.get("trailingPegRatio"),
            ev_ebitda=info.get("enterpriseToEbitda"),
            ps=info.get("priceToSalesTrailing12Months"),
            pb=info.get("priceToBook"),
            div_yield=(info.get("dividendYield") or 0) * 100,
            pct52=pct52,
            verdict=verdict,
            emoji=emoji,
            score=score,
            signals=signals,
            info=info,
            is_etf=is_etf,
        ))

if not rows:
    st.error("所有代码均无法获取数据")
    st.stop()

# ── group rows by sector ───────────────────────────────────────────────────────

from collections import defaultdict as _dd
sector_rows = _dd(list)
for r in rows:
    if r["is_etf"]:
        sector_rows["ETF"].append(r)
    else:
        en = r["info"].get("sector", "其他")
        sector_rows[SECTOR_CN.get(en, en)].append(r)

# sort rows by sector
sorted_rows = [r for s in sorted(sector_rows) for r in sector_rows[s]]

# ── overview cards ─────────────────────────────────────────────────────────────

n = len(sorted_rows)
cols = st.columns(min(n, 5))
for i, r in enumerate(sorted_rows):
    with cols[i % 5]:
        pe_str  = fmt(r["trailing_pe"])
        pct_str = f"{r['pct52']*100:.0f}%" if r["pct52"] is not None else "—"
        st.metric(
            label=f"{r['emoji']} **{r['sym']}**",
            value=f"${r['price']:.2f}",
            delta=f"{r['verdict']}  |  P/E {pe_str}  |  52w {pct_str}",
            delta_color="off",
        )

st.divider()

# ── detail table ───────────────────────────────────────────────────────────────

st.subheader("指标总览")

table = []
for r in sorted_rows:
    sn = next(s for s in sector_rows if r in sector_rows[s])
    table.append({
        "代码":        r["sym"],
        "名称":        r["name"],
        "板块":        sn,
        "现价":        f"${r['price']:.2f}",
        "P/E":         fmt(r["trailing_pe"]),
        "Forward P/E": fmt(r["forward_pe"]),
        "PEG":         fmt(r["peg"], 2),
        "EV/EBITDA":   fmt(r["ev_ebitda"]),
        "P/S":         fmt(r["ps"]),
        "P/B":         fmt(r["pb"]),
        "股息率":      fmt(r["div_yield"], 2, "%") if r["div_yield"] else "—",
        "52周位置":    f"{r['pct52']*100:.0f}%" if r["pct52"] is not None else "—",
        "估值判断":    f"{r['emoji']} {r['verdict']}",
        "得分":        f"{r['score']:+.1f}",
    })

st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

st.divider()

# ── per-stock detail ───────────────────────────────────────────────────────────

st.subheader("逐只详情")

for r in sorted_rows:
    with st.expander(f"{r['emoji']}  {r['sym']} — {r['verdict']}（得分 {r['score']:+.1f}）"):
        left, right = st.columns([1, 2])

        with left:
            st.markdown(f"**{r['name']}**")
            st.caption(next(s for s in sector_rows if r in sector_rows[s]))

            if r["pct52"] is not None:
                st.write(f"**52周区间位置：{r['pct52']*100:.0f}%**")
                st.progress(float(r["pct52"]))
                lo = r["info"].get("fiftyTwoWeekLow", 0)
                hi = r["info"].get("fiftyTwoWeekHigh", 0)
                a, b, c = st.columns(3)
                a.caption(f"低 ${lo:.2f}")
                b.caption(f"现 ${r['price']:.2f}")
                c.caption(f"高 ${hi:.2f}")

            st.divider()
            st.write("**估值信号**")
            if r["signals"]:
                for sig in r["signals"]:
                    st.write(f"· {sig}")
            else:
                st.caption("数据不足")

        with right:
            price_chart(r["sym"], r["info"])

import streamlit as st
import yfinance as yf
import google.generativeai as genai
import os
from dotenv import load_dotenv
import uuid
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, time as dt_time, timedelta
import requests
import xml.etree.ElementTree as ET
import time
from database import init_db, get_user, create_user, update_token, check_monthly_reset, add_comment, get_comments, login_user_google

# ==========================================
# 1. SYSTEM CONFIG
# ==========================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
init_db()

st.set_page_config(
    page_title="SahamWajar Pro",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CRITICAL FIX: SESSION STATE INIT (MUST BE AT THE TOP) ---
if 'is_logged_in' not in st.session_state: st.session_state.is_logged_in = False
if 'user_info' not in st.session_state: st.session_state.user_info = {"name": "Guest", "email": None, "avatar": None}
if 'target_ticker' not in st.session_state: st.session_state.target_ticker = None
if 'chart_range' not in st.session_state: st.session_state.chart_range = '1D'
if 'reply_to' not in st.session_state: st.session_state.reply_to = None
if 'ai_cache' not in st.session_state: st.session_state.ai_cache = {}

# Security
fingerprint_script = """<script src="https://cdn.jsdelivr.net/npm/@fingerprintjs/fingerprintjs@3/dist/fp.min.js"></script><script>function sendDeviceId() {FingerprintJS.load().then(fp => {fp.get().then(result => {const urlParams = new URLSearchParams(window.location.search);if (!urlParams.has('device_id')) {urlParams.set('device_id', result.visitorId);window.location.search = urlParams.toString();}});});}window.addEventListener('load', sendDeviceId);</script>"""
st.components.v1.html(fingerprint_script, height=0, width=0)

query_params = st.query_params
device_id = query_params.get("device_id", "TEMP_DEV")

# Load User Logic
token_sisa = 3
if device_id != "TEMP_DEV":
    user_db = get_user(device_id)
    if user_db:
        token_sisa = user_db[4]
        if user_db[1]: # Jika email ada di database
            st.session_state.is_logged_in = True
            st.session_state.user_info = {"name": user_db[2], "email": user_db[1], "avatar": user_db[3]}
    else:
        create_user(device_id)

@st.cache_data(ttl=300)
def fetch_market_data_cached(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="2y"), stock.info, stock.balance_sheet

# ==========================================
# 2. UI STYLES
# ==========================================
def fmt_rp(val, short=False):
    if val is None: return "-"
    if short:
        if abs(val) >= 1e12: return f"{val/1e12:.2f}T"
        elif abs(val) >= 1e9: return f"{val/1e9:.2f}M"
    return f"Rp {val:,.0f}"

def card(l, v, s=""): 
    return f"""<div class="stat-card"><div class="stat-label">{l}</div><div class="stat-val">{v}</div><div style="font-size:0.7rem; color:#58a6ff;">{s}</div></div>"""

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
    .stApp { background-color: #0e1117; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    
    /* CARDS */
    .stat-card { background: #161b22; border: 1px solid #30363d; padding: 12px; border-radius: 6px; text-align: center; margin-bottom: 10px; }
    .stat-label { color: #8b949e; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; }
    .stat-val { color: #fff; font-size: 1.1rem; font-weight: 700; margin-top: 4px; }
    
    /* NEWS FEED */
    .news-scroll { max-height: 550px; overflow-y: auto; padding: 10px; background: #161b22; border: 1px solid #30363d; border-top: none; border-radius: 0 0 8px 8px; }
    .news-header { margin-bottom: 0px; padding-bottom: 10px; border-bottom: 1px solid #30363d; font-weight: bold; color: white; }
    .news-item { background: transparent; border-bottom: none; padding: 8px 0; text-decoration: none; display: block; margin-bottom: 8px; transition: 0.1s; }
    .news-item:hover .news-tit { color: #4facfe; }
    .news-src { font-size: 0.6rem; color: #4facfe; font-weight: 700; margin-bottom: 2px; display:block;}
    .news-tit { color: #e6edf3; font-weight: 500; font-size: 0.85rem; line-height: 1.4; transition: color 0.2s;}
    .news-date { font-size: 0.65rem; color: #8b949e; margin-top: 2px; display:block;}
    
    /* COMMENTS */
    .comment-container { margin-top: 20px; padding-top: 20px; border-top: 1px solid #30363d; }
    .comment-box { background: #161b22; border-radius: 8px; padding: 12px; margin-bottom: 10px; border: 1px solid #21262d; }
    .comment-user { font-weight: 700; color: #4facfe; font-size: 0.85rem; }
    .comment-time { font-size: 0.7rem; color: #8b949e; }
    .comment-text { font-size: 0.9rem; color: #e6edf3; line-height: 1.4; margin-top: 5px;}
    
    /* AI BOX */
    .ai-box { background: radial-gradient(circle at top left, #1f2937 0%, #0f172a 100%); border: 1px solid #334155; border-left: 4px solid #8b5cf6; border-radius: 12px; padding: 20px; margin-top: 20px; margin-bottom: 20px; }

    /* SKELETON */
    @keyframes shimmer { 0% { background-position: -1000px 0; } 100% { background-position: 1000px 0; } }
    .skeleton-box { background: #1f2937; border-radius: 8px; padding: 15px; margin-bottom: 10px; border: 1px solid #30363d; }
    .skeleton-line { height: 10px; background: #2d3446; background-image: linear-gradient(to right, #2d3446 0%, #3f4b61 20%, #2d3446 40%, #2d3446 100%); background-repeat: no-repeat; background-size: 1000px 100%; animation: shimmer 1.5s infinite linear forwards; border-radius: 4px; margin-bottom: 8px; }

    /* BUTTONS */
    .stockbit-btn { background: transparent; border: 1px solid #2d3446; color: #eaecef; padding: 8px; border-radius: 6px; text-decoration: none; font-weight: 600; font-size: 0.9rem; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 8px; transition: 0.2s; }
    .stockbit-btn:hover { background: rgba(59, 130, 246, 0.1); border-color: #3b82f6; color: #3b82f6; }
    div[data-testid="stPills"] button[aria-selected="true"] { background-color: #238636 !important; color: white !important; }
    
    .profile-card { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; display: flex; align-items: center; gap: 10px; margin-bottom: 20px; }
    .profile-avatar { width: 40px; height: 40px; border-radius: 50%; background: #3b82f6; display:flex; align-items:center; justify-content:center; font-weight:bold; color:white;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. CORE ENGINES
# ==========================================
def get_market_status_wib():
    utc_now = datetime.utcnow(); wib_now = utc_now + timedelta(hours=7); curr = wib_now.time(); day = wib_now.weekday()
    if day > 4: return "CLOSED", "#ef4444", "🔴"
    if day == 4: 
        if dt_time(11, 30) <= curr <= dt_time(14, 0): return "ISTIRAHAT", "#f59e0b", "☕"
    else: 
        if dt_time(12, 0) <= curr <= dt_time(13, 30): return "ISTIRAHAT", "#f59e0b", "☕"
    if dt_time(9, 0) <= curr <= dt_time(16, 0): return "OPEN", "#22c55e", "🟢"
    return "CLOSED", "#ef4444", "🔴"

class TechnicalEngine:
    @staticmethod
    def add_indicators(df):
        if df.empty: return df
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean(); rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

class ChartEngine:
    @staticmethod
    def get_chart_data(ticker, time_range):
        stock = yf.Ticker(ticker)
        params = {'1D': {'period': '1d', 'interval': '1m'}, '1W': {'period': '5d', 'interval': '5m'}, '1M': {'period': '1mo', 'interval': '60m'}, '3M': {'period': '3mo', 'interval': '1d'}, 'YTD': {'period': 'ytd', 'interval': '1d'}, '1Y': {'period': '1y', 'interval': '1d'}, '3Y': {'period': '3y', 'interval': '1wk'}, '5Y': {'period': '5y', 'interval': '1wk'}, 'ALL': {'period': 'max', 'interval': '1mo'}}
        p = params.get(time_range, params['1D'])
        try:
            hist = stock.history(period=p['period'], interval=p['interval'])
            if hist.empty: return pd.DataFrame()
            if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
            hist.index = pd.to_datetime(hist.index)
            if hist.index.tz is None: hist.index = hist.index.tz_localize('UTC').tz_convert('Asia/Jakarta')
            else: hist.index = hist.index.tz_convert('Asia/Jakarta')
            hist = TechnicalEngine.add_indicators(hist)
            return hist
        except: return pd.DataFrame()

class NewsEngine:
    @staticmethod
    def get_google_news(query, limit=8):
        try:
            clean = query.replace(" ", "%20")
            url = f"https://news.google.com/rss/search?q={clean}&hl=id-ID&gl=ID&ceid=ID:id"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=3)
            news_list = []
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall('./channel/item')[:limit]:
                    title = item.find('title').text
                    link = item.find('link').text
                    source = item.find('source').text if item.find('source') is not None else "News"
                    try: dt = datetime.strptime(item.find('pubDate').text, "%a, %d %b %Y %H:%M:%S %Z"); d_str = dt.strftime("%d %b %H:%M")
                    except: d_str = "Terkini"
                    news_list.append({"title": title, "link": link, "publisher": source, "date": d_str})
            return news_list
        except: return []
    @staticmethod
    def get_latest_news(ticker, name):
        t1 = NewsEngine.get_google_news(ticker, limit=5)
        clean = name.replace("Tbk", "").replace("PT", "").replace("Persero", "").strip()
        t2 = NewsEngine.get_google_news(clean, limit=5)
        seen = set(); final_obj = []; final_txt = []
        for x in t1 + t2:
            if x['link'] not in seen: final_obj.append(x); seen.add(x['link'])
        return final_obj[:10]

class FinancialEngine:
    @staticmethod
    def get_realtime_currency():
        try: return yf.Ticker("IDR=X").history(period="1d")['Close'].iloc[-1]
        except: return 16000.0
    @staticmethod
    def get_fundamental_data(ticker):
        try:
            hist_raw, info, bs = fetch_market_data_cached(ticker)
            if hist_raw.empty: return None
            hist = hist_raw.copy()
            if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
            hist.index = pd.to_datetime(hist.index)
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else curr
            chg_pct = ((curr - prev) / prev) * 100
            eps = info.get('trailingEps', 0)
            bvps = info.get('bookValue', 0)
            if (bvps is None or bvps == 0):
                pbv = info.get('priceToBook', 0)
                if pbv and pbv > 0: bvps = curr / pbv
            kurs = FinancialEngine.get_realtime_currency()
            safe_eps = float(eps) if eps else 0.0
            safe_bvps = float(bvps) if bvps else 0.0
            is_conv = False
            if curr > 500:
                if abs(safe_eps) < 50 and abs(safe_eps) > 0 and (curr/safe_eps > 100): safe_eps *= kurs; is_conv = True
                if abs(safe_bvps) < 50 and abs(safe_bvps) > 0 and (curr/safe_bvps > 100): safe_bvps *= kurs; is_conv = True
            per = (curr/safe_eps) if safe_eps > 0 else 0
            pbv = (curr/safe_bvps) if safe_bvps > 0 else 0
            return { "ticker": ticker, "name": info.get('longName', ticker), "sector": info.get('sector', '-'), "industry": info.get('industry', '-'), "price": curr, "change_pct": chg_pct, "eps": safe_eps, "bvps": safe_bvps, "per": per, "pbv": pbv, "mkt_cap": info.get('marketCap', 0), "kurs_val": kurs, "is_converted": is_conv }
        except: return None
    @staticmethod
    def calculate_graham(eps, bvps):
        if eps > 0 and bvps > 0: return (22.5 * eps * bvps)**0.5
        return 0

# ==========================================
# 4. RENDER FUNCTIONS
# ==========================================
def render_skeleton_news():
    html = ""
    for _ in range(6): html += """<div class="skeleton-box"><div class="skeleton-line" style="width:60%"></div><div class="skeleton-line" style="width:40%; height:8px;"></div></div>"""
    return html

def render_skeleton_ai():
    return """<div class="ai-box" style="min-height:200px;"><div class="skeleton-line" style="width:40%; margin-bottom:15px;"></div><div class="skeleton-line" style="width:100%"></div><div class="skeleton-line" style="width:95%"></div><div class="skeleton-line" style="width:90%"></div><div class="skeleton-line" style="width:60%"></div></div>"""

def render_chart(hist, chart_type="Line", height=500, time_range="1D"):
    if hist.empty: return None
    min_y = hist['Low'].min(); max_y = hist['High'].max()
    padding = (max_y - min_y) * 0.05 if (max_y - min_y) > 0 else max_y * 0.01
    start_p = hist['Close'].iloc[0]; end_p = hist['Close'].iloc[-1]
    t_color = '#22c55e' if end_p >= start_p else '#ef4444'
    f_color = 'rgba(34, 197, 94, 0.1)' if end_p >= start_p else 'rgba(239, 68, 68, 0.1)'
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])
    if chart_type == "Line": fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', line=dict(color=t_color, width=2), fill='tozeroy', fillcolor=f_color, name='Price'), row=1, col=1)
    else:
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='OHLC', increasing_line_color='#22c55e', decreasing_line_color='#ef4444'), row=1, col=1)
        if len(hist) > 50: fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='#eab308', width=1), name='MA50'), row=1, col=1)
    vol_c = [t_color if c >= o else '#ef4444' for o, c in zip(hist['Open'], hist['Close'])]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=vol_c, name='Vol'), row=2, col=1)
    if time_range == "1D":
        today_date = hist.index[0].date()
        is_friday = today_date.weekday() == 4
        break_start = datetime.combine(today_date, dt_time(11, 30) if is_friday else dt_time(12, 0)).replace(tzinfo=hist.index.tz)
        break_end = datetime.combine(today_date, dt_time(14, 0) if is_friday else dt_time(13, 30)).replace(tzinfo=hist.index.tz)
        try:
            fig.add_vrect(x0=break_start, x1=break_end, fillcolor="rgba(50, 50, 50, 0.5)", layer="below", line_width=0, row=1, col=1)
            mid_time = break_start + (break_end - break_start)/2
            fig.add_annotation(x=mid_time, y=max_y, text="☕ ISTIRAHAT", showarrow=False, font=dict(size=10, color="gray"), bgcolor="rgba(0,0,0,0.7)", yanchor="top", row=1, col=1)
        except: pass
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#131722', xaxis_rangeslider_visible=False, showlegend=False, hovermode="x unified")
    fig.update_yaxes(range=[min_y - padding, max_y + padding], gridcolor='rgba(255,255,255,0.05)', side='right', row=1, col=1)
    fig.update_yaxes(showticklabels=False, gridcolor='rgba(255,255,255,0.05)', row=2, col=1)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 🦅 SahamWajar")
    if st.session_state.is_logged_in:
        st.markdown(f"""<div class="profile-card"><div class="profile-avatar">{st.session_state.user_info['name'][0]}</div><div class="profile-info"><div class="profile-name">{st.session_state.user_info['name']}</div><div class="profile-status">VIP MEMBER</div></div></div>""", unsafe_allow_html=True)
    else:
        if st.button("🔑 Login (Simulasi)"):
            login_user_google(device_id, "user@gmail.com", "Investor Sultan", "S")
            st.session_state.is_logged_in = True; st.session_state.user_info = {"name": "Investor Sultan", "email": "user@gmail.com", "avatar": "S"}; st.rerun()
    
    st.markdown("---")
    
    # LOGIC RESET SEARCH
    def on_search():
        if st.session_state.search_key:
            st.session_state.target_ticker = st.session_state.search_key.upper()
            st.session_state.chart_range = '1D'
            # Clear AI Cache if new ticker
            if 'ai_cache' in st.session_state:
                if st.session_state.target_ticker in st.session_state.ai_cache:
                    del st.session_state.ai_cache[st.session_state.target_ticker]

    target_input = st.text_input("Kode Saham", placeholder="BBRI, ADRO...", label_visibility="collapsed", key="search_key", on_change=on_search).upper()
    
    if st.button("🔎 ANALISA", type="primary", use_container_width=True):
        if target_input: on_search(); st.rerun()

    if st.button("🏠 IHSG DASHBOARD", use_container_width=True): st.session_state.target_ticker = None; st.rerun()
    
    st.markdown("---")
    if st.session_state.target_ticker:
        st.markdown(f"""<a href="https://stockbit.com/symbol/{st.session_state.target_ticker}" target="_blank" class="stockbit-btn"><img src="https://connect-assets.prosple.com/cdn/ff/sLrGtYss5TXkr7reuS4ZwlbYrRPGt2Pn5KqBOoIi2wQ/1633922595/public/styles/scale_and_crop_center_120x120/public/2021-10/logo%20stockbit.png?itok=Yww-Qnl2" width="18" height="18" style="border-radius:2px;"> Buka Stockbit</a>""", unsafe_allow_html=True)
        st.link_button("📊 Google Finance", f"https://www.google.com/finance/quote/{st.session_state.target_ticker}:IDX", use_container_width=True)
    m_msg, m_col, m_icon = get_market_status_wib()
    st.markdown(f"<div style='margin-top:10px; text-align:center; color:{m_col}; border:1px solid {m_col}; padding:6px; border-radius:6px; font-weight:bold; font-size:0.8rem;'>{m_icon} {m_msg}</div>", unsafe_allow_html=True)
    st.caption(f"Token: {token_sisa if not st.session_state.is_logged_in else '∞'}")

# ==========================================
# 6. MAIN CONTROLLER
# ==========================================
target = st.session_state.target_ticker

if not target:
    st.title("Index Harga Saham Gabungan (IHSG) 🇮🇩")
    col_L, col_R = st.columns([3, 1])
    with col_L:
        c1, c2 = st.columns([3, 1])
        with c1: t_range = st.pills("Range", ['1D', '1W', '1M', 'YTD', '1Y', '5Y'], default='1D', key="ihsg_range")
        with c2: c_style = st.pills("Style", ['Line', 'Candle'], default='Line', key="ihsg_style")
        with st.spinner("Loading..."):
            hist = ChartEngine.get_chart_data("^JKSE", t_range)
            data = FinancialEngine.get_fundamental_data("^JKSE")
            if not hist.empty and data:
                st.plotly_chart(render_chart(hist, c_style, height=450, time_range=t_range), use_container_width=True)
                m1, m2, m3 = st.columns(3)
                chg = data['change_pct']; col = "green" if chg >= 0 else "red"
                m1.metric("IHSG Level", f"{data['price']:,.2f}", f"{chg:.2f}%")
                m2.metric("Vol", fmt_rp(hist['Volume'].iloc[-1], short=True))
                m3.metric("Status", m_msg)
    with col_R:
        st.markdown('<div class="news-header"><b>📰 Makro News</b></div>', unsafe_allow_html=True)
        news_ph = st.empty()
        news_ph.markdown(render_skeleton_news(), unsafe_allow_html=True)
        news_list = NewsEngine.get_google_news("Ekonomi Indonesia IHSG", limit=8)
        news_html = '<div class="news-scroll">'
        if news_list:
            for n in news_list: news_html += f"""<a href="{n['link']}" target="_blank" class="news-item"><span class="news-src">{n['publisher']}</span><div class="news-tit">{n['title']}</div><span class="news-date">{n['date']}</span></a>"""
        else: news_html += "<div style='padding:10px; color:gray'>Tidak ada berita.</div>"
        news_html += '</div>'
        news_ph.markdown(news_html, unsafe_allow_html=True)

else:
    if token_sisa <= 0 and not st.session_state.is_logged_in: st.error("Kuota Habis.")
    else:
        header_ph = st.empty()
        col_main, col_news = st.columns([2.8, 1.2])
        chart_ph = col_main.empty()
        metrics_ph = col_main.empty()
        ai_ph = col_main.empty()
        comment_ph = col_main.empty()
        news_side_ph = col_news.empty()

        with header_ph: st.markdown(f"## ⏳ Loading {target}...")
        
        # RENDER SKELETONS
        # Cek Cache AI dulu
        if 'ai_cache' not in st.session_state: st.session_state.ai_cache = {}
        if target not in st.session_state.ai_cache:
            with ai_ph: st.markdown(render_skeleton_ai(), unsafe_allow_html=True)
        else:
             with ai_ph: st.markdown(f"""<div class="ai-box"><h4 style="margin:0 0 10px 0; color:#22c55e">🧠 AI Strategy</h4><div style="color:#d4d4d8; font-size:0.9rem; line-height:1.6;">{st.session_state.ai_cache[target]}</div></div>""", unsafe_allow_html=True)
             
        with news_side_ph.container():
            st.markdown('<div class="news-header"><b>📰 Berita Saham</b></div>', unsafe_allow_html=True)
            st.markdown(render_skeleton_news(), unsafe_allow_html=True)

        data = FinancialEngine.get_fundamental_data(f"{target}.JK")
        if not data: st.error("Data Tidak Ditemukan.")
        else:
            if device_id != "TEMP_DEV" and not st.session_state.is_logged_in: update_token(device_id, -1)
            chg = data['change_pct']; cc = "#22c55e" if chg >= 0 else "#ef4444"
            with header_ph.container():
                st.markdown(f"""<div style="display:flex; justify-content:space-between; align-items:center;"><div><h2 style="margin:0;">{data['name']}</h2><div style="color:#848e9c;">{data['ticker']} • {data['sector']}</div></div><div style="text-align:right;"><div style="font-size:2rem; font-weight:700;">{fmt_rp(data['price'])}</div><div style="color:{cc}; font-weight:600;">{chg:+.2f}%</div></div></div>""", unsafe_allow_html=True)
                if data['is_converted']: st.info(f"ℹ️ USD Converted (Rate: {fmt_rp(data['kurs_val'])})")

            with chart_ph.container():
                r1, r2 = st.columns([3, 1])
                with r1: t_range = st.pills("Timeframe", ['1D', '1W', '1M', '3M', 'YTD', '1Y', '3Y', '5Y'], default='1D', key="stock_range")
                with r2: c_style = st.pills("Mode", ['Line', 'Candle'], default='Line', key="stock_type")
                hist_dyn = ChartEngine.get_chart_data(f"{target}.JK", t_range)
                if not hist_dyn.empty: st.plotly_chart(render_chart(hist_dyn, c_style, height=500, time_range=t_range), use_container_width=True)

            graham = FinancialEngine.calculate_graham(data['eps'], data['bvps'])
            status = "FAIR"; gc = "grey"
            if graham > 0:
                status = "UNDERVALUED" if data['price'] < graham else "OVERVALUED"
                gc = "#22c55e" if status == "UNDERVALUED" else "#ef4444"

            with metrics_ph.container():
                st.markdown("### 📊 Key Stats")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(card("Market Cap", fmt_rp(data['mkt_cap'], short=True)), unsafe_allow_html=True)
                c2.markdown(card("EPS", fmt_rp(data['eps'])), unsafe_allow_html=True)
                c3.markdown(card("PER", f"{data['per']:.2f}x"), unsafe_allow_html=True)
                c4.markdown(card("PBV", f"{data['pbv']:.2f}x"), unsafe_allow_html=True)
                c5.markdown(f"""<div class="stat-card" style="border-color:{gc}"><div class="stat-label">GRAHAM</div><div class="stat-val" style="color:{gc}">{fmt_rp(graham)}</div><div style="font-size:0.7rem; color:{gc}">{status}</div></div>""", unsafe_allow_html=True)

            # RENDER NEWS (Async)
            news_list = NewsEngine.get_latest_news(target, data['name'])
            with news_side_ph.container():
                st.markdown('<div class="news-header"><b>📰 Berita Saham</b></div>', unsafe_allow_html=True)
                news_html = '<div class="news-scroll">'
                if news_list:
                    for n in news_list: news_html += f"""<a href="{n['link']}" target="_blank" class="news-item"><span class="news-src">{n['publisher']}</span><div class="news-tit">{n['title']}</div><span class="news-date">{n['date']}</span></a>"""
                else: news_html += "<div style='padding:10px; color:gray'>Tidak ada berita.</div>"
                news_html += '</div>'
                news_side_ph.markdown(news_html, unsafe_allow_html=True)

            # RENDER COMMENTS (INSTANT - NO WAIT)
            with comment_ph.container():
                st.markdown("---")
                st.markdown(f"### 💬 Diskusi {target}")
                with st.form(key=f'comment_form_{target}'):
                    user_comment = st.text_area("Tulis pendapat Anda...", height=80)
                    submit_comment = st.form_submit_button("Kirim Komentar")
                    if submit_comment:
                        if st.session_state.is_logged_in:
                            add_comment(target, device_id, st.session_state.user_info['name'], st.session_state.user_info['name'][0], user_comment)
                            st.success("Terkirim!")
                            time.sleep(0.1)
                            st.rerun()
                        else: st.warning("Silakan Login.")
                comments = get_comments(target)
                if comments:
                    for c in comments:
                        is_reply = c['parent_id'] is not None
                        indent = "margin-left:40px; border-left:3px solid #30363d;" if is_reply else ""
                        bg = "#0d1117" if is_reply else "#161b22"
                        st.markdown(f"""<div style="background:{bg}; border-radius:8px; padding:15px; margin-bottom:10px; border:1px solid #21262d; {indent}"><div class="c-head"><div class="c-av">{c['avatar']}</div><div class="c-user">{c['username']}</div><div class="c-time">{c['timestamp']}</div></div><div class="c-text">{c['content']}</div></div>""", unsafe_allow_html=True)
                        if not is_reply and st.session_state.is_logged_in:
                            if st.button("↪ Balas", key=f"rep_{c['id']}"):
                                st.session_state.reply_to = c['id']
                                st.rerun()
                        if st.session_state.reply_to == c['id']:
                            with st.form(key=f"form_rep_{c['id']}"):
                                txt = st.text_input(f"Balas {c['username']}...")
                                if st.form_submit_button("Kirim"):
                                    add_comment(target, device_id, st.session_state.user_info['name'], st.session_state.user_info['name'][0], txt, c['id'])
                                    st.session_state.reply_to = None
                                    st.rerun()
                else: st.caption("Belum ada diskusi.")

            # AI (ONLY IF NOT CACHED)
            if target not in st.session_state.ai_cache:
                rsi_now = hist_dyn['RSI'].iloc[-1] if not hist_dyn.empty else 0
                news_titles = [n['title'] for n in news_list]
                n_ctx = "\n".join(news_titles) if news_titles else "-"
                prompt = f"Analisa Saham {target}. Price {data['price']}, Graham {graham} ({status}). Timeframe {t_range}, RSI {rsi_now:.1f}. News: {n_ctx}. Strategi?"
                model = genai.GenerativeModel('gemini-2.5-flash')
                res = model.generate_content(prompt)
                st.session_state.ai_cache[target] = res.text
                with ai_ph.container():
                    st.markdown(f"""<div class="ai-box"><h4 style="margin:0 0 10px 0; color:#22c55e">🧠 AI Strategy</h4><div style="color:#d4d4d8; font-size:0.9rem; line-height:1.6;">{res.text}</div></div>""", unsafe_allow_html=True)
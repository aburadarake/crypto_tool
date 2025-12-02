import streamlit as st
import google.generativeai as genai
from duckduckgo_search import DDGS
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime

# --- ページ設定 ---
st.set_page_config(page_title="TATSUO GOD MODE", page_icon="🦁", layout="wide")

# ==========================================
# 🔐 セキュリティ: ライセンスキー認証
# ==========================================
def check_password():
    if "auth_status" not in st.session_state:
        st.session_state["auth_status"] = False
    if st.session_state["auth_status"]:
        return True

    st.markdown("""
    <style>
    /* ▼▼▼ 全フォント統一 (念押し) ▼▼▼ */
    * { font-family: 'Arial', sans-serif !important; }
    
    .stApp { background-color: #0b0e11; color: #e1e1e1; }
    .login-box {
        background-color: #1e2329; padding: 40px; border-radius: 10px;
        border: 1px solid #F0B90B; text-align: center; margin-top: 80px;
        box-shadow: 0 0 30px rgba(240, 185, 11, 0.15);
    }
    h1 { color: #F0B90B; letter-spacing: 2px; }
    .stTextInput input { 
        background-color: #2a2e39 !important; color: white !important; 
        border: 1px solid #4a4e59 !important; text-align: center; letter-spacing: 3px;
    }
    .stButton>button { 
        background: linear-gradient(90deg, #F0B90B, #D8A000); 
        color: black; font-weight: 800; width: 100%; border: none;
        padding: 10px; font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<div class='login-box'><h1>🦁 TATSUO PRO</h1><p>MEMBER LOGIN</p></div>", unsafe_allow_html=True)
        input_key = st.text_input("ENTER LICENSE KEY", type="password", key="login_pw")
        
        if st.button("LOGIN"):
            try:
                valid_keys = st.secrets["passwords"]["valid_keys"]
                if input_key in valid_keys:
                    st.session_state["auth_status"] = True
                    st.rerun()
                else: st.error("⛔ Invalid Key")
            except:
                if input_key == "god_mode":
                    st.session_state["auth_status"] = True
                    st.rerun()
                else: st.error("Authentication Failed")
    return False

if not check_password(): st.stop()

# ==========================================
# 💾 ログ管理システム
# ==========================================
if 'history' not in st.session_state:
    st.session_state['history'] = []

def save_log(symbol, price, sentiment, risk_mode, analysis_text):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "price": price,
        "sentiment": sentiment,
        "risk_mode": risk_mode,
        "full_text": analysis_text
    }
    st.session_state['history'].insert(0, log_entry)

def get_last_analysis(symbol):
    for log in st.session_state['history']:
        if log['symbol'] == symbol:
            return log
    return None

# ==========================================
# 🦁 以下、ツール本体
# ==========================================

# --- 🎨 UIデザイン ---
st.markdown("""
<style>
    /* ▼▼▼ ここでも念押しでフォント統一 ▼▼▼ */
    * { font-family: 'Arial', sans-serif !important; }

    .stApp { background-color: #0b0e11; color: #e1e1e1; }
    section[data-testid="stSidebar"] { background-color: #15191f; border-right: 1px solid #2a2e39; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    .stTextInput input, .stNumberInput input { background-color: #2a2e39 !important; color: #ffffff !important; border: 1px solid #4a4e59 !important; }
    input[type="password"] { background-color: #2a2e39 !important; color: #ffffff !important; }
    .metric-card { background-color: #1e2329; padding: 15px; border-radius: 8px; border: 1px solid #2a2e39; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 10px; }
    div[data-testid="stMetricValue"] { font-size: 26px !important; color: #fff !important; }
    div[data-testid="stMetricLabel"] { color: #848e9c !important; font-size: 14px !important; }
    .stButton>button { background: linear-gradient(135deg, #F0B90B 0%, #D8A000 100%); color: #000; font-weight: 800; border: none; height: 3.5em; border-radius: 4px; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(240, 185, 11, 0.4); }
    
    /* ▼▼▼ AIボックスにもフォント指定を追加 ▼▼▼ */
    .ai-box { 
        background-color: #161a1e; border: 1px solid #2a2e39; border-left: 6px solid #F0B90B; 
        padding: 30px; border-radius: 8px; margin-top: 20px; color: #e1e1e1; line-height: 1.8;
        font-family: 'Arial', sans-serif !important; 
    }
    
    .ai-box h1 { color: #F0B90B; font-size: 28px; border-bottom: 2px solid #2a2e39; padding-bottom: 15px; margin-bottom: 20px; }
    .ai-box h2 { color: #0ECB81; font-size: 22px; margin-top: 30px; margin-bottom: 15px; border-left: 4px solid #0ECB81; padding-left: 10px; }
    .ai-box th { background-color: #2b313a; color: #F0B90B; padding: 12px; border: 1px solid #363c4e; text-align: left; }
    .ai-box td { border: 1px solid #363c4e; padding: 12px; color: #e1e1e1; }
    .ai-box strong { color: #F0B90B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- サイドバー設定 ---
st.sidebar.title("🦁 TATSUO GOD MODE")
st.sidebar.caption("Ver.22.1 Font Fixed")
st.sidebar.markdown("---")

user_name = st.sidebar.text_input("お名前", value="ユーザー")
api_key = st.sidebar.text_input("Google API Key", type="password", help="ご自身のキーを入力してください")
if api_key: genai.configure(api_key=api_key)

st.sidebar.markdown("### 💰 資金・リスク設定")
usdt_margin = st.sidebar.number_input("証拠金 (USDT)", value=1000.0, step=100.0)
leverage = st.sidebar.select_slider("レバレッジ (倍)", options=[1, 5, 10, 20, 50, 100, 500], value=10)

risk_mode = st.sidebar.select_slider("トレードスタイル", options=["安全重視 🛡️", "バランス ⚖️", "積極運用 🔥"], value="バランス ⚖️")

if risk_mode == "安全重視 🛡️": risk_percent, sl_mult = 0.01, 1.5
elif risk_mode == "積極運用 🔥": risk_percent, sl_mult = 0.03, 3.0
else: risk_percent, sl_mult = 0.02, 2.0

power = usdt_margin * leverage
st.sidebar.success(f"💥 運用パワー: ${power:,.0f}")

# --- ログ表示 ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📜 Log")
if st.sidebar.button("ログをCSVで保存"):
    if st.session_state['history']:
        df_log = pd.DataFrame(st.session_state['history'])
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Download CSV", csv, "tatsuo_log.csv", "text/csv")
    else:
        st.sidebar.warning("履歴なし")

# --- 関数群 (API・計算) ---
def get_fear_greed():
    try:
        url = "https://api.alternative.me/fng/"
        resp = requests.get(url).json()
        return int(resp['data'][0]['value']), resp['data'][0]['value_classification']
    except: return 50, "Neutral"

def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    flow = typical * df['Volume']
    pos_flow = flow.where(typical > typical.shift(1), 0).rolling(14).sum()
    neg_flow = flow.where(typical < typical.shift(1), 0).rolling(14).sum()
    mfi_ratio = pos_flow / neg_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_ma'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_up'] = df['BB_ma'] + (df['BB_std'] * 2)
    df['BB_low'] = df['BB_ma'] - (df['BB_std'] * 2)

    high_9 = df['High'].rolling(9).max()
    low_9 = df['Low'].rolling(9).min()
    tenkan = (high_9 + low_9) / 2
    high_26 = df['High'].rolling(26).max()
    low_26 = df['Low'].rolling(26).min()
    kijun = (high_26 + low_26) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    high_52 = df['High'].rolling(52).max()
    low_52 = df['Low'].rolling(52).min()
    span_b = ((high_52 + low_52) / 2).shift(26)
    
    df['SpanA'] = span_a
    df['SpanB'] = span_b
    df['Kijun'] = kijun
    return df

def get_market_data(symbol):
    ticker = f"{symbol.upper()}-USD" if not symbol.endswith("-USD") else symbol.upper()
    try:
        df = yf.download(ticker, period="1y", interval="1d")
        if df.empty: return None, None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = calculate_indicators(df)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        last_6mo = df.tail(180)
        h = last_6mo['High'].max()
        l = last_6mo['Low'].min()
        diff = h - l
        fibs = {
            "0.236": h - diff * 0.236, "0.382": h - diff * 0.382,
            "0.500": h - diff * 0.500, "0.618": h - diff * 0.618,
            "High": h, "Low": l
        }
        pivot = (prev['High'] + prev['Low'] + prev['Close']) / 3

        data = {
            'Close': latest['Close'], 'RSI': latest['RSI'], 'MFI': latest['MFI'],
            'ATR': latest['ATR'], 'SMA_200': latest['SMA_200'],
            'Cloud_Top': max(latest['SpanA'], latest['SpanB']),
            'Cloud_Bottom': min(latest['SpanA'], latest['SpanB']),
            'Kijun': latest['Kijun'],
            'MACD': latest['MACD'], 'Signal': latest['Signal'],
            'BB_up': latest['BB_up'], 'BB_low': latest['BB_low'],
            'Fib_High': h, 'Fib_Low': l, 
            'Fib_0618': h - diff * 0.618, 'Fib_0382': h - diff * 0.382,
            'Pivot': pivot, 
            'Change': (latest['Close'] - prev['Close']) / prev['Close'] * 100,
            'Volume': latest['Volume']
        }
        return df, data
    except: return None, None

def plot_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', increasing_line_color='#0ECB81', decreasing_line_color='#F6465D'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='#F0B90B', width=1), name='SMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#FFFFFF', width=1.5), name='SMA 200'))
    fig.update_layout(template='plotly_dark', plot_bgcolor='#161A25', paper_bgcolor='#161A25', height=550,
        margin=dict(l=0, r=0, t=40, b=0), xaxis_rangeslider_visible=False,
        title=dict(text=f"{symbol}/USDT Professional Chart", font=dict(color="#EAECEF", size=18)))
    return fig

def search_news(keywords):
    try:
        with DDGS() as ddgs:
            res = [r['title'] for r in ddgs.text(f"{keywords} crypto news analysis", max_results=3)]
            return "\n".join(res)
    except: return "No News"

# --- メイン処理 ---
c1, c2 = st.columns([3, 1])
with c1: symbol_input = st.text_input("SYMBOL", value="BTC", label_visibility="collapsed")
with c2: analyze = st.button("🚀 神分析を実行")

if analyze:
    if not api_key: st.warning("⚠️ API Keyを入れてください")
    else:
        with st.spinner(f"🦁 {user_name}さんのために徹底分析中..."):
            hist_df, data = get_market_data(symbol_input)
            fg_val, fg_class = get_fear_greed()
        
        if data is None: st.error("データ取得エラー")
        else:
            cp = float(data['Close'])
            bp = usdt_margin * leverage
            liq = cp * (1 - (1/leverage))
            
            # 過去ログ取得
            last_log = get_last_analysis(symbol_input)
            last_log_text = ""
            if last_log:
                last_log_text = f"""
                【前回の分析ログ ({last_log['timestamp']})】
                - 当時価格: ${last_log['price']:,.2f}
                - 判断: {last_log['sentiment']}
                - 内容: {last_log['full_text'][:300]}...
                ※ この過去データと比較し、シナリオの変化と前回の正誤を検証して精度を上げよ。
                """

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Current Price", f"${cp:,.2f}", f"{data['Change']:.2f}%")
            fg_col = "#0ECB81" if fg_val > 75 else "#F6465D" if fg_val < 25 else "#F0B90B"
            with m2: st.markdown(f'<div class="metric-card"><div style="color:#848e9c;font-size:12px;">Fear & Greed</div><div style="color:{fg_col};font-size:24px;font-weight:bold;">{fg_val} ({fg_class})</div></div>', unsafe_allow_html=True)
            mfi_col = "#0ECB81" if data['MFI'] > 80 else "#F6465D" if data['MFI'] < 20 else "#fff"
            with m3: st.markdown(f'<div class="metric-card"><div style="color:#848e9c;font-size:12px;">Money Flow (MFI)</div><div style="color:{mfi_col};font-size:24px;font-weight:bold;">{data["MFI"]:.1f}</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="metric-card"><div style="color:#848e9c;font-size:12px;">Est. Liquidation</div><div style="color:#F6465D;font-size:24px;font-weight:bold;">${liq:,.2f}</div></div>', unsafe_allow_html=True)

            st.plotly_chart(plot_chart(hist_df, symbol_input), use_container_width=True)

            news = search_news(symbol_input)
            trend = "上昇" if cp > data['SMA_200'] else "下落"
            cloud = "雲上(強気)" if cp > data['Cloud_Top'] else "雲下(弱気)" if cp < data['Cloud_Bottom'] else "雲中(レンジ)"
            fib_st = "レンジ内"
            
            fib_0618 = data.get('Fib_0618', 0)
            fib_0382 = data.get('Fib_0382', 0)

            if cp > fib_0618: fib_st = "0.618突破(強気)"
            elif cp < fib_0382: fib_st = "0.382割れ(弱気)"

            macd_st = "ゴールデンクロス(買い)" if data['MACD'] > data['Signal'] else "デッドクロス(売り)"
            bb_st = "バンドウォーク警戒" if cp > data['BB_up'] else "売られすぎ" if cp < data['BB_low'] else "レンジ内"

            sl_dist = data['ATR'] * sl_mult
            sl_long = cp - sl_dist
            sl_short = cp + sl_dist
            risk_amt = usdt_margin * risk_percent
            rec_lot = risk_amt / sl_dist

            prompt = f"""
            あなたは伝説の相場師「バフェット達夫 (Ver.22.1 Font Fixed)」です。
            ユーザーの「{user_name}さん」に対し、Gensparkを超える熱量と論理で分析せよ。
            
            {last_log_text}

            【今回の分析対象: {symbol_input}】
            - 現在価格: ${cp:,.2f} (24h変動: {data['Change']:.2f}%)
            
            【テクニカル指標】
            1. RSI(14): {data['RSI']:.2f}
            2. MACD: {macd_st}
            3. BB: {bb_st}
            4. 雲: {cloud} (基準線 ${data['Kijun']:.2f})
            5. MFI: {data['MFI']:.1f}
            6. センチメント: {fg_val} ({fg_class})

            【{user_name}さんの資金設定: {risk_mode}】
            - 運用パワー: ${bp:,.0f} (レバ{leverage}倍)
            - 許容リスク: 資金の{risk_percent*100}% (${risk_amt:.2f})

            【最新ニュース】
            {news}

            ==================================================
            【必須出力フォーマット (Markdown)】
            
            # 🔥 バフェット達夫の神判断: [LONG / SHORT / WAIT] (確度 S/A/B)

            {user_name}さん、お待たせしました。{risk_mode}での分析結果です。
            **(結論を一言で)**

            ## 🔄 前回分析との比較 (Self-Correction)
            **(前回ログがある場合、ここになぜ予想が当たったか/外れたか、何が変わったかを記述。ない場合は「初回分析です」と記述)**

            ## 🎯 現在の市場状況 (3点チェック)
            | 項目 | 数値/状態 | 達夫の判定 |
            | :--- | :--- | :--- |
            | **需給 (MFI)** | {data['MFI']:.1f} | (判定) |
            | **構造 (Fib)** | {fib_st} | (判定) |
            | **トレンド** | {trend} / {cloud} | (判定) |

            ## 🚀 戦略シナリオ (3つの分岐)
            ### 🔥 シナリオA：メイン戦略 (確率 60%)
            * **展開:** ...
            * **目標:** ...
            * **期待利益:** +$XXX (概算)

            ### ⚠️ シナリオB：調整/トラップ (確率 30%)
            * **展開:** ...

            ### 🚨 シナリオC：緊急撤退 (確率 10%)
            * **撤退ライン:** ${sl_long:.2f} (Long) / ${sl_short:.2f} (Short)

            ## 💰 推奨エントリー指示書 (厳守)
            * **推奨アクション:** [成行エントリー / 指値待ち / 様子見]
            * **適正ロット数:** 約 **{rec_lot:.4f} {symbol_input}**
            * **損切り(SL):** **${sl_long:.2f} (Long) / ${sl_short:.2f} (Short)**
            * **利確(TP):** 第1目標 ${data['Pivot']:.2f}, 最終目標 ${data['Fib_High']:.2f}

            ## 🦁 達夫からの熱いメッセージ
            (最後に熱いアドバイス)
            """

            with st.spinner("🦁 シナリオ構築中..."):
                try:
                    # gemini-1.5-flashに固定 (最も安定・高速)
                    model = genai.GenerativeModel('gemini-2.5-pro')
                    response = model.generate_content(prompt)
                    ai_text = response.text
                    
                    st.markdown(f'<div class="ai-box">{ai_text}</div>', unsafe_allow_html=True)
                    
                    # ログ保存
                    sentiment = "WAIT"
                    if "LONG" in ai_text.split("\n")[0]: sentiment = "LONG"
                    elif "SHORT" in ai_text.split("\n")[0]: sentiment = "SHORT"
                    
                    save_log(symbol_input, cp, sentiment, risk_mode, ai_text)
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# --- 履歴表示エリア (メイン画面下部) ---
if st.session_state['history']:
    st.markdown("---")
    st.subheader("📜 過去の分析ログ (ブラウザを閉じると消えます)")
    for log in st.session_state['history']:
        with st.expander(f"{log['timestamp']} - {log['symbol']} ({log['sentiment']})"):
            st.markdown(log['full_text'])

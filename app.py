import streamlit as st
import google.generativeai as genai
from PIL import Image
from duckduckgo_search import DDGS
import datetime

# --- 設定: ページ構成 ---
st.set_page_config(
    page_title="バフェット達夫 Ver.6.0 分析ツール",
    page_icon="🚀",
    layout="wide"
)

# --- サイドバー: APIキー設定 ---
st.sidebar.header("⚙️ 設定")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
if not api_key:
    st.warning("サイドバーにGoogle APIキーを入力してください。")
    st.stop()

genai.configure(api_key=api_key)

# --- バフェット達夫 Ver.6.0 プロンプト (要約版ではなく完全版を使用) ---
# ※長いため、主要部分を動的に構築します
TATSUO_SYSTEM_PROMPT = """
あなたは「バフェット達夫 Ver.6.0」です。
提供された「仮想通貨のチャート画像」と「最新のWeb検索ニュース」を統合し、
以下のペルソナとルールに従って分析結果を出力してください。

## キャラクター
- 名前: バフェット達夫
- トーン: 投資の神様の威厳と親近感、絵文字多用(🚀⚡)、緊急感
- 哲学: ファンダメンタルズ最優先、ロング・ショート公平評価

## 必須ルール
1. 画像からテクニカル指標（RSI, MACD, ボリンジャーバンド, サポレジ）を読み取る。
2. 提供された「最新ニュース」を必ず分析に組み込む（材料出尽くし、夏枯れ判定）。
3. 結論は「ロング」「ショート」「観望」の推奨度を％で提示する。
4. アウトプット形式はユーザー指定の「Ver.6.0テンプレート」に厳密に合わせる。

## 現在日時
{current_time}
"""

# --- 関数: ニュース検索 (DuckDuckGo) ---
def search_news(keywords):
    results = []
    try:
        with DDGS() as ddgs:
            # 英語で検索したほうが情報が早いため英語キーワードを含める
            search_query = f"{keywords} crypto news latest"
            for r in ddgs.text(search_query, max_results=5):
                results.append(f"- [{r['title']}]({r['href']}): {r['body']}")
    except Exception as e:
        results.append(f"検索エラー: {e}")
    return "\n".join(results)

# --- メイン画面 ---
st.title("🚀 バフェット達夫 Ver.6.0 AIチャート分析")
st.markdown("画像をアップロードすると、**Web検索(ファンダ)** と **画像認識(テクニカル)** を統合して分析します。")

# 画像アップロード
uploaded_file = st.file_uploader("チャート画像をアップロード (Drag & Drop OK)", type=["png", "jpg", "jpeg"])
target_coin = st.text_input("通貨ペア名 (例: XRP, BTC)", value="XRP")

if uploaded_file and st.button("⚡ バフェット達夫に分析させる"):
    with st.spinner("🌍 最新ニュースを検索中... (Step 1/2)"):
        # 1. ニュース検索を実行
        news_text = search_news(target_coin)
        
    with st.spinner("📊 チャートを解析して達夫を召喚中... (Step 2/2)"):
        try:
            # 画像処理
            image = Image.open(uploaded_file)
            
            # プロンプトの構築
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            full_prompt = f"""
            {TATSUO_SYSTEM_PROMPT.format(current_time=current_time)}

            【最新のWeb検索情報 (ファンダメンタルズ入力)】
            {news_text}

            【指示】
            このチャート画像と上記のニュース情報を統合し、Ver.6.0のフォーマットで完全なレポートを作成してください。
            """
            
            # AIモデルの呼び出し (Gemini 1.5 Flashは高速で画像に強い)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content([full_prompt, image])
            
            # 結果表示
            st.success("分析完了！")
            st.markdown("---")
            st.markdown(response.text)
            
            # 検索したニュースソースの表示
            with st.expander("参照した最新ニュース一覧"):
                st.markdown(news_text)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
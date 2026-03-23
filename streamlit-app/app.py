import streamlit as st
import psycopg2
import psycopg2.extras
import requests
import os
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="NewsPersona", page_icon="📰", layout="wide")
DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "postgres"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB",   "optimize"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASS", "password"),
}

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://ollama:11434") + "/api/chat"
OLLAMA_MODEL = "qwen2:7b"

TOPIC_LABELS = {
    "Cong Nghe": "Công Nghệ",
    "Thoi Su":   "Thời Sự",
    "Phap Luat": "Pháp Luật",
    "The Thao":  "Thể Thao",
    "Giao Duc":  "Giáo Dục",
    "Kinh Te":   "Kinh Tế",
}

# ─────────────────────────────────────────────
# PERSONA CONFIG
# ─────────────────────────────────────────────
PERSONA_CONFIG = {
    "Chuyên gia Công nghệ": {
        "topics": ["Cong Nghe"],
        "desc": "Phân tích kỹ thuật & xu hướng tương lai",
        "prompt": (
            "Bạn là một CTO dày dạn kinh nghiệm. "
            "Hãy phân tích tin tức dưới góc độ kỹ thuật, giải pháp công nghệ và xu hướng tương lai. "
            "Trả lời chuyên sâu, dùng thuật ngữ ngành khi cần thiết."
        ),
    },
    "Nhà phân tích Kinh tế": {
        "topics": ["Kinh Te"],
        "desc": "Chỉ số thị trường, rủi ro & tác động vĩ mô",
        "prompt": (
            "Bạn là chuyên gia kinh tế trưởng. "
            "Hãy tập trung vào các con số, chỉ số thị trường, rủi ro tài chính và tác động vĩ mô của tin tức. "
            "Trả lời có số liệu cụ thể và nhận định rõ ràng."
        ),
    },
    "Phóng viên Thời sự": {
        "topics": ["Thoi Su", "Phap Luat"],
        "desc": "Ngắn gọn, khách quan, đúng trọng tâm",
        "prompt": (
            "Bạn là một phóng viên hiện trường kỳ cựu. "
            "Hãy tóm tắt tin tức cực kỳ ngắn gọn, súc tích và khách quan. "
            "Nhấn mạnh vào: Ai, Cái gì, Ở đâu, Khi nào, Tại sao."
        ),
    },
    "Blogger Thể thao": {
        "topics": ["The Thao", "Giao Duc"],
        "desc": "Năng động, hào hứng, gần gũi",
        "prompt": (
            "Bạn là một Influencer thể thao năng động. "
            "Hãy kể lại tin tức bằng giọng văn hóm hỉnh, tập trung vào cảm xúc và trải nghiệm. "
            "Dùng ngôn ngữ trẻ trung, gần gũi với người đọc."
        ),
    },
    "Độc giả Tổng hợp": {
        "topics": [],
        "desc": "Cân bằng, đa chiều, dễ đọc",
        "prompt": (
            "Bạn là trợ lý tin tức thân thiện. "
            "Hãy trình bày thông tin cân bằng, khách quan và dễ hiểu. "
            "Đưa ra đủ các khía cạnh quan trọng mà không thiên vị."
        ),
    },
}

DEFAULT_SYSTEM = "Bạn là trợ lý tin tức. Hãy trả lời câu hỏi của người dùng một cách ngắn gọn và khách quan."

def assign_persona(selected_topics: list) -> str:
    best, best_score = "Độc giả Tổng hợp", 0
    for name, cfg in PERSONA_CONFIG.items():
        if name == "Độc giả Tổng hợp":
            continue
        score = len(set(selected_topics) & set(cfg["topics"]))
        if score > best_score:
            best, best_score = name, score
    return best

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
@st.cache_resource

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def fetch_articles(topics: list, limit: int = 30):
    get_conn.clear()
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, url, topic, published_at, content
                FROM public.raw_data
                WHERE topic = ANY(%s)
                ORDER BY published_at DESC
                LIMIT %s
                """,
                (topics, limit),
            )
            return cur.fetchall()
    except Exception as e:
        conn.rollback()  # ← reset transaction bị kẹt
        raise e

# ─────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────
def call_ollama(system_prompt: str, user_message: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"❌ Lỗi kết nối Ollama: {e}"

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=IBM+Plex+Sans:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.article-card {
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
    background: #fff;
}
.article-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: #1e293b;
    line-height: 1.45;
    margin: 4px 0 0 0;
}
.article-meta { font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px; }
.badge {
    display: inline-block;
    background: #eff6ff;
    color: #2563eb;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 6px;
}
.persona-banner {
    background: linear-gradient(135deg, #1e293b, #334155);
    border-radius: 10px;
    padding: 18px 24px;
    color: white;
    margin: 12px 0 20px 0;
}
.persona-banner h3 { margin: 0 0 4px 0; font-size: 1.3rem; color: white; }
.persona-banner p  { margin: 0; opacity: .75; font-size: 0.88rem; }

.box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 16px;
    min-height: 160px;
    white-space: pre-wrap;
}
.box-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Cá nhân hóa")
    st.markdown("Chọn **3–4 chủ đề** bạn quan tâm:")

    user_interests = st.multiselect(
        label="Chủ đề",
        options=list(TOPIC_LABELS.keys()),
        format_func=lambda x: TOPIC_LABELS[x],
        max_selections=4,
        label_visibility="collapsed",
    )

    n = len(user_interests)
    if n < 3:
        st.info(f"Chọn thêm {3 - n} chủ đề nữa.")
        active_persona = "Độc giả Tổng hợp"
    else:
        active_persona = assign_persona(user_interests)
        cfg = PERSONA_CONFIG[active_persona]
        #st.success(f"Persona: **{cfg['icon']} {active_persona}**")
        st.caption(cfg["desc"])

    st.markdown("---")
    st.caption(f"Model: `{OLLAMA_MODEL}`")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
persona_cfg = PERSONA_CONFIG[active_persona]

st.markdown("# NewsPersona")
st.markdown(
    f"""
    <div class="persona-banner">
        <p>{persona_cfg['desc']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_feed, tab_chat = st.tabs(["📋 Feed bài báo", "💬 Hỏi đáp & So sánh"])

# ══════════════════════════════════════════════
# TAB 1 — FEED
# ══════════════════════════════════════════════
with tab_feed:
    if not user_interests:
        st.info("👈 Chọn chủ đề ở sidebar để xem bài báo.")
    else:
        try:
            articles = fetch_articles(user_interests)
        except Exception as e:
            st.error(f"Lỗi kết nối database: {e}")
            articles = []

        if not articles:
            st.warning("Chưa có bài báo nào cho các chủ đề đã chọn.")
        else:
            st.markdown(f"**{len(articles)} bài báo mới nhất**")

            for art in articles:
                pub = art["published_at"]
                pub_str = pub.strftime("%d/%m/%Y %H:%M") if isinstance(pub, datetime) else str(pub)[:16]

                st.markdown(
                    f"""
                    <div class="article-card">
                        <div class="article-meta">
                            <span class="badge">{TOPIC_LABELS.get(art['topic'], art['topic'])}</span>
                            {pub_str}
                        </div>
                        <div class="article-title">
                            <a href="{art['url']}" target="_blank" style="color:inherit;text-decoration:none;">
                                {art['title']}
                            </a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("🔍 Tóm tắt có / không có Persona Prompt"):
                    if st.button("▶ Tạo tóm tắt", key=f"btn_{art['id']}"):
                        snippet  = art["content"][:2500]
                        user_msg = f"Tóm tắt bài báo sau trong 3–4 câu bằng tiếng Việt:\n\n{snippet}"

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="box-label" style="color:#64748b;">⬜ Không có Persona Prompt</div>', unsafe_allow_html=True)
                            with st.spinner("Đang tạo..."):
                                res_no = call_ollama(DEFAULT_SYSTEM, user_msg)
                            st.markdown(f'<div class="box">{res_no}</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="box-label" style="color:#2563eb;">🎯 Persona: {active_persona}</div>', unsafe_allow_html=True)
                            with st.spinner("Đang tạo..."):
                                res_yes = call_ollama(persona_cfg["prompt"], user_msg)
                            st.markdown(f'<div class="box">{res_yes}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — CHAT & SO SÁNH
# ══════════════════════════════════════════════
with tab_chat:
    st.markdown("Đặt câu hỏi về tin tức — xem AI trả lời **có** và **không có** Persona Prompt.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Hiển thị lịch sử chat
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="box-label" style="color:#64748b;">⬜ Không có Prompt</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="box">{msg["no_persona"]}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="box-label" style="color:#2563eb;">🎯 {active_persona}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="box">{msg["with_persona"]}</div>', unsafe_allow_html=True)
            st.markdown("")

    # Chat input
    user_input = st.chat_input("Nhập câu hỏi của bạn...")
    if user_input:
        # Hiện tin nhắn user
        with st.chat_message("user"):
            st.markdown(user_input)

        # Gọi Ollama 2 lần song song
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="box-label" style="color:#64748b;">⬜ Không có Persona Prompt</div>', unsafe_allow_html=True)
            with st.spinner("Đang trả lời..."):
                res_no = call_ollama(DEFAULT_SYSTEM, user_input)
            st.markdown(f'<div class="box">{res_no}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="box-label" style="color:#2563eb;">🎯 Persona: {active_persona}</div>', unsafe_allow_html=True)
            with st.spinner("Đang trả lời..."):
                res_yes = call_ollama(persona_cfg["prompt"], user_input)
            st.markdown(f'<div class="box">{res_yes}</div>', unsafe_allow_html=True)

        # Lưu vào history
        st.session_state.chat_history.append({"role": "user",      "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "no_persona": res_no, "with_persona": res_yes})

    if st.session_state.chat_history:
        if st.button("🗑️ Xóa lịch sử"):
            st.session_state.chat_history = []
            st.rerun()
import streamlit as st
import psycopg2
import psycopg2.extras
import requests
import os
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="NewsPersona", page_icon="📰", layout="wide")

DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "postgres"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB",   "optimize"),
    "user":     os.getenv("PG_USER", "kietcorn"),
    "password": os.getenv("PG_PASS", "kiietqo9204"),
}

OLLAMA_BASE  = os.getenv("OLLAMA_URL", "http://10.4.21.3:11435")
OLLAMA_URL   = f"{OLLAMA_BASE}/api/chat"
EMBED_URL    = f"{OLLAMA_BASE}/api/embeddings"
OLLAMA_MODEL = "qwen3.5:4b"
EMBED_MODEL  = "nomic-embed-text:latest"

TOPIC_LABELS = {
    "Cong Nghe": "Công Nghệ",
    "Thoi Su":   "Thời Sự",
    "Phap Luat": "Pháp Luật",
    "The Thao":  "Thể Thao",
    "Giao Duc":  "Giáo Dục",
    "Kinh Te":   "Kinh Tế",
}

PERSONA_CONFIG = {
    "Chuyên gia Công nghệ": {
        "topics":       ["Cong Nghe"],
        "desc":         "Phân tích kỹ thuật & xu hướng tương lai",
        "prompt":       "Bạn là một CTO dày dạn kinh nghiệm. Phân tích tin tức dưới góc độ kỹ thuật, xu hướng công nghệ và tác động thực tiễn. Trả lời chuyên sâu, súc tích.",
    },
    "Nhà phân tích Kinh tế": {
        "topics":       ["Kinh Te"],
        "desc":         "Chỉ số thị trường, rủi ro & tác động vĩ mô",
        "prompt":       "Bạn là chuyên gia kinh tế trưởng. Tập trung vào số liệu, chỉ số thị trường, rủi ro tài chính và tác động vĩ mô. Trả lời có số liệu cụ thể.",
    },
    "Phóng viên Thời sự": {
        "topics":       ["Thoi Su", "Phap Luat"],
        "desc":         "Ngắn gọn, khách quan, đúng trọng tâm 5W1H",
        "prompt":       "Bạn là phóng viên hiện trường kỳ cựu. Tóm tắt tin tức ngắn gọn, khách quan. Nhấn mạnh: Ai, Cái gì, Ở đâu, Khi nào, Tại sao.",
    },
    "Blogger Thể thao": {
        "topics":       ["The Thao", "Giao Duc"],
        "desc":         "Năng động, hào hứng, gần gũi",
        "prompt":       "Bạn là Influencer thể thao năng động. Kể lại tin tức bằng giọng hóm hỉnh, tập trung vào cảm xúc. Dùng ngôn ngữ trẻ trung, gần gũi.",
    },
    "Độc giả Tổng hợp": {
        "topics":       [],
        "desc":         "Cân bằng, đa chiều, dễ đọc",
        "prompt":       "Bạn là trợ lý tin tức thân thiện. Trình bày thông tin cân bằng, khách quan, dễ hiểu. Đưa ra đủ các khía cạnh quan trọng.",
    },
}

DEFAULT_PROMPT = "Bạn là trợ lý tin tức. Trả lời ngắn gọn và khách quan."

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
# SESSION STATE
# ─────────────────────────────────────────────
if "read_article_ids" not in st.session_state:
    st.session_state.read_article_ids = []      # id bài đã đọc
if "read_vectors" not in st.session_state:
    st.session_state.read_vectors = []          # embedding bài đã đọc
if "user_vector" not in st.session_state:
    st.session_state.user_vector = None         # vector sở thích hiện tại
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_persona" not in st.session_state:
    st.session_state.active_persona = "Độc giả Tổng hợp"

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
@st.cache_resource
def get_conn():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        return conn
    except Exception as e:
        st.error(f"❌ Lỗi kết nối Database: {e}")
        return None

def fetch_data(query, params=None):
    get_conn.clear()
    conn = get_conn()
    if not conn:
        return []
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall()
    except Exception as e:
        st.error(f"❌ Lỗi truy vấn: {e}")
        return []

# JOIN raw_data + n8n_vector qua title matching (đã xác nhận hoạt động)
JOIN_CLAUSE = """
    FROM raw_data r
    JOIN n8n_vectors v
        ON v.text LIKE '%%' || LEFT(r.title, 30) || '%%'
"""

def fetch_articles_by_topics(topics: list, limit: int = 15):
    placeholders = ", ".join(["%s"] * len(topics))
    query = f"""
        SELECT DISTINCT ON (r.id)
            r.id, r.title, r.url, r.topic, r.published_at, r.content,
            v.id as vector_id, v.embedding
        {JOIN_CLAUSE}
        WHERE r.topic IN ({placeholders})
        ORDER BY r.id, r.published_at DESC
        LIMIT %s
    """
    return fetch_data(query, tuple(topics) + (limit,))

def fetch_latest_articles(limit: int = 15):
    query = f"""
        SELECT DISTINCT ON (r.id)
            r.id, r.title, r.url, r.topic, r.published_at, r.content,
            v.id as vector_id, v.embedding
        {JOIN_CLAUSE}
        ORDER BY r.id, r.published_at DESC
        LIMIT %s
    """
    return fetch_data(query, (limit,))

def fetch_similar_articles(user_vec: list, limit: int = 10):
    """Vector search: tìm bài gần nhất với user vector"""
    query = f"""
        SELECT DISTINCT ON (r.id)
            r.id, r.title, r.url, r.topic, r.published_at, r.content,
            v.embedding,
            1 - (v.embedding <=> %s::vector) AS score
        {JOIN_CLAUSE}
        ORDER BY r.id, v.embedding <=> %s::vector ASC
        LIMIT %s
    """
    return fetch_data(query, (user_vec, user_vec, limit))

def fetch_rag_context(query_vec: list, top_k: int = 3):
    """Lấy top_k bài liên quan nhất để đưa vào RAG context"""
    query = f"""
        SELECT DISTINCT ON (r.id)
            r.title, r.topic,
            LEFT(r.content, 500) as content_snippet,
            1 - (v.embedding <=> %s::vector) AS score
        {JOIN_CLAUSE}
        ORDER BY r.id, v.embedding <=> %s::vector ASC
        LIMIT %s
    """
    return fetch_data(query, (query_vec, query_vec, top_k))

# ─────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────
def embed_text(text: str) -> list | None:
    """Embed text bằng nomic-embed-text qua Ollama"""
    try:
        r = requests.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=70,
        )
        r.raise_for_status()
        return r.json()["embedding"]
    except Exception as e:
        st.warning(f"⚠️ Lỗi embed: {e}")
        return None

def ask_ollama(system_prompt: str, user_msg: str) -> str:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            "stream": False,
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"⚠️ Lỗi Ollama: {e}"

def ask_ollama_rag(system_prompt: str, context_articles: list, user_question: str) -> str:
    """Gọi Ollama với RAG context từ bài báo liên quan"""
    if not context_articles:
        return ask_ollama(system_prompt, user_question)

    context_text = "\n\n".join([
        f"[{i+1}] {a['title']} ({TOPIC_LABELS.get(a['topic'], a['topic'])}):\n{a['content_snippet']}"
        for i, a in enumerate(context_articles)
    ])

    rag_system = f"""{system_prompt}

Dưới đây là các bài báo liên quan để tham khảo khi trả lời:
{context_text}

Hãy trả lời dựa trên các bài báo trên. Nếu câu hỏi không liên quan đến bài báo, hãy trả lời theo kiến thức chung."""

    return ask_ollama(rag_system, user_question)

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
    padding: 16px 22px;
    color: white;
    margin-bottom: 20px;
}
.persona-banner h3 { margin: 0 0 4px 0; font-size: 1.2rem; color: white; }
.persona-banner p  { margin: 0; opacity: .75; font-size: 0.85rem; }
.compare-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px;
    min-height: 140px;
    white-space: pre-wrap;
}
.compare-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.rag-source {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.8rem;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📰 NewsPersona")
    st.markdown("Chọn **3–4 chủ đề** bạn quan tâm:")

    interests = st.multiselect(
        label="topics",
        options=list(TOPIC_LABELS.keys()),
        format_func=lambda x: TOPIC_LABELS[x],
        max_selections=4,
        label_visibility="collapsed",
    )

    if interests:
        # Gán persona dựa trên topics đã chọn
        persona_name = assign_persona(interests)
        st.session_state.active_persona = persona_name
        cfg = PERSONA_CONFIG[persona_name]
        st.success(f"Persona: ** {persona_name}**")
        st.caption(cfg["desc"])

        # Tính user_vector = trung bình embedding các bài thuộc topics đã chọn
        if st.button("🔄 Cập nhật Feed", use_container_width=True):
            with st.spinner("Đang tính vector sở thích..."):
                rows = fetch_articles_by_topics(interests, limit=100)
                if rows:
                    vecs = []
                    for row in rows:
                        emb = row.get("embedding")
                        if emb is not None:
                            if isinstance(emb, str):
                                import ast
                                emb = ast.literal_eval(emb)
                            vecs.append(np.array(emb, dtype=np.float32))
                    if vecs:
                        st.session_state.user_vector = np.mean(vecs, axis=0).tolist()
                        st.success(f"Feed đã cập nhật từ {len(vecs)} bài!")

    st.markdown("---")

    # Hiển thị số bài đã đọc
    n_read = len(st.session_state.read_article_ids)
    if n_read > 0:
        st.metric("Bài đã đọc", n_read)
        if st.button("🔄 Cập nhật Feed từ hành vi", use_container_width=True):
            if st.session_state.read_vectors:
                combined = np.mean(st.session_state.read_vectors, axis=0).tolist()
                # Blend 50/50 giữa topic vector và behavior vector
                if st.session_state.user_vector:
                    topic_vec  = np.array(st.session_state.user_vector)
                    behav_vec  = np.array(combined)
                    st.session_state.user_vector = ((topic_vec + behav_vec) / 2).tolist()
                else:
                    st.session_state.user_vector = combined
                st.success("Feed đã thích nghi theo hành vi đọc!")

    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"Model: `{OLLAMA_MODEL}`")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
persona_cfg  = PERSONA_CONFIG[st.session_state.active_persona]
persona_name = st.session_state.active_persona

st.markdown("# 📰 NewsPersona")
st.markdown(
    f"""<div class="persona-banner">
        <h3>{persona_name}</h3>
        <p>{persona_cfg['desc']}</p>
    </div>""",
    unsafe_allow_html=True,
)

tab_feed, tab_latest, tab_chat = st.tabs([
    "✨ Feed For You",
    "🗞️ Tất cả bài báo",
    "💬 Hỏi đáp (RAG)",
])

# ─────────────────────────────────────────────
def render_article(art, key_prefix: str, show_score: bool = False):
    pub = art.get("published_at")
    pub_str = pub.strftime("%d/%m/%Y %H:%M") if isinstance(pub, datetime) else str(pub)[:16]
    score_str = f" | Độ phù hợp: {art['score']:.0%}" if show_score and "score" in art else ""

    st.markdown(
        f"""<div class="article-card">
            <div class="article-meta">
                <span class="badge">{TOPIC_LABELS.get(art['topic'], art['topic'])}</span>
                {pub_str}{score_str}
            </div>
            <div class="article-title">
                <a href="{art['url']}" target="_blank" style="color:inherit;text-decoration:none;">
                    {art['title']}
                </a>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        already_read = str(art["id"]) in st.session_state.read_article_ids
        if not already_read:
            if st.button("✅ Đã đọc", key=f"read_{key_prefix}_{art['id']}"):
                st.session_state.read_article_ids.append(str(art["id"]))
                emb = art.get("embedding")
                if emb is not None:
                    if isinstance(emb, str):
                        import ast
                        emb = ast.literal_eval(emb)
                    st.session_state.read_vectors.append(np.array(emb, dtype=np.float32))
                st.toast("Đã ghi nhận!")
                st.rerun()
        else:
            st.caption("✅ Đã đọc")

    with st.expander("🔍 Tóm tắt có / không có Persona Prompt"):
        if st.button("▶ Tạo tóm tắt", key=f"sum_{key_prefix}_{art['id']}"):
            snippet  = art.get("content", "")[:2000]
            user_msg = f"Tóm tắt bài báo sau trong 3–4 câu bằng tiếng Việt:\n\n{snippet}"

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="compare-label" style="color:#64748b;">⬜ Không có Persona Prompt</div>', unsafe_allow_html=True)
                with st.spinner("Đang tạo..."):
                    r1 = ask_ollama(DEFAULT_PROMPT, user_msg)
                st.markdown(f'<div class="compare-box">{r1}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="compare-label" style="color:#2563eb;">🎯 {persona_name}</div>', unsafe_allow_html=True)
                with st.spinner("Đang tạo..."):
                    r2 = ask_ollama(persona_cfg["prompt"], user_msg)
                st.markdown(f'<div class="compare-box">{r2}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 1 — FEED CÁ NHÂN
# ══════════════════════════════════════════════
with tab_feed:
    if st.session_state.user_vector is None:
        st.info("👈 Chọn chủ đề ở sidebar rồi bấm **Cập nhật Feed** để xem gợi ý cá nhân.")
    else:
        st.markdown(f"**Gợi ý dựa trên sở thích của bạn** {'(đã cập nhật theo hành vi)' if st.session_state.read_vectors else ''}")
        articles = fetch_similar_articles(st.session_state.user_vector, limit=10)
        if not articles:
            st.warning("Chưa tìm được bài phù hợp.")
        else:
            for art in articles:
                render_article(art, key_prefix="feed", show_score=True)

# ══════════════════════════════════════════════
# TAB 2 — TẤT CẢ BÀI BÁO
# ══════════════════════════════════════════════
with tab_latest:
    st.markdown("**Tất cả bài báo mới nhất**")
    latest = fetch_latest_articles(limit=20)
    if not latest:
        st.warning("Chưa có bài báo nào.")
    else:
        for art in latest:
            render_article(art, key_prefix="latest")

# ══════════════════════════════════════════════
# TAB 3 — CHAT RAG
# ══════════════════════════════════════════════
with tab_chat:
    st.markdown(f"Hỏi về tin tức — AI tìm bài liên quan rồi trả lời theo **Persona {persona_name}**")
    st.markdown("---")

    # Hiển thị lịch sử
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["answer"])
                if msg.get("sources"):
                    with st.expander("📚 Nguồn bài báo được dùng"):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<div class="rag-source">📄 <b>{src["title"]}</b> '
                                f'<span style="color:#94a3b8">— {TOPIC_LABELS.get(src["topic"], src["topic"])}</span></div>',
                                unsafe_allow_html=True,
                            )

    # Input
    user_input = st.chat_input("Hỏi về tin tức...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm bài liên quan..."):
                # Embed câu hỏi
                query_vec = embed_text(user_input)

            sources = []
            if query_vec:
                with st.spinner("Đang tìm kiếm..."):
                    rag_articles = fetch_rag_context(query_vec, top_k=3)
                    sources = [dict(a) for a in rag_articles] if rag_articles else []

            with st.spinner("Đang trả lời..."):
                answer = ask_ollama_rag(persona_cfg["prompt"], sources, user_input)

            st.markdown(answer)

            if sources:
                with st.expander("📚 Nguồn bài báo được dùng"):
                    for src in sources:
                        st.markdown(
                            f'<div class="rag-source">📄 <b>{src["title"]}</b> '
                            f'<span style="color:#94a3b8">— {TOPIC_LABELS.get(src["topic"], src["topic"])}</span></div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.chat_history.append({"role": "user",      "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "answer": answer, "sources": sources})

    if st.session_state.chat_history:
        if st.button("🗑️ Xóa lịch sử chat"):
            st.session_state.chat_history = []
            st.rerun()
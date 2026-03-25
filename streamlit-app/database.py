"""
persona_clustering.py
─────────────────────
Phân cụm Persona từ embedding bài báo, tự động đặt tên bằng Ollama.

Quy trình:
  1. Kéo embedding + title từ PostgreSQL (JOIN n8n_vector + raw_data)
  2. Elbow Method + Silhouette Score (Rousseeuw, 1987) → tìm k tối ưu
  3. K-Means clustering (MacQueen, 1967)
  4. Ollama đọc top bài mỗi cụm → tự đặt tên + viết prompt
  5. Lưu persona_config.json → dùng trong app.py

Cài đặt:
  pip install psycopg2-binary scikit-learn numpy matplotlib requests

Chạy:
  python persona_clustering.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import psycopg2.extras
import requests
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB",   "optimize"),
    "user":     os.getenv("PG_USER", "kietcorn"),
    "password": os.getenv("PG_PASS", "kiietqo9204"),
}

OLLAMA_BASE  = os.getenv("OLLAMA_URL", "http://10.4.21.3:11435")
OLLAMA_URL   = f"{OLLAMA_BASE}/api/chat"
OLLAMA_MODEL = "qwen2.5:4b"

K_MIN = 2    # số cụm tối thiểu cần thử
K_MAX = 10   # số cụm tối đa cần thử
TOP_N = 5    # số bài đại diện mỗi cụm để đặt tên persona

# ─────────────────────────────────────────────
# STEP 1: KÉO DỮ LIỆU
# ─────────────────────────────────────────────
def fetch_embeddings():
    print("⏳ Đang kết nối Database và tải dữ liệu...")
    conn = psycopg2.connect(**DB_CONFIG)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT ON (r.id)
                r.id,
                r.title,
                r.topic,
                v.embedding::text AS embedding
            FROM raw_data r
            JOIN n8n_vectors v
                ON v.text LIKE '%' || LEFT(r.title, 30) || '%'
            WHERE v.embedding IS NOT NULL
            ORDER BY r.id
        """)
        rows = cur.fetchall()

    conn.close()

    ids, titles, topics, vectors = [], [], [], []
    for row in rows:
        ids.append(str(row["id"]))
        titles.append(row["title"])
        topics.append(row["topic"])
        vectors.append(json.loads(row["embedding"]))

    X = np.array(vectors, dtype=np.float32)
    print(f"✅ Đã tải {len(X)} bài báo, vector {X.shape[1]} chiều.")
    return ids, titles, topics, X


# ─────────────────────────────────────────────
# STEP 2: TÌM K TỐI ƯU
# ─────────────────────────────────────────────
def find_optimal_k(X: np.ndarray) -> int:
    print(f"\n🔍 Tìm số cụm tối ưu (K={K_MIN}..{K_MAX})...")
    print(f"{'K':>4} | {'WCSS':>14} | {'Silhouette':>10}")
    print("-" * 36)

    wcss_list, sil_list = [], []
    k_range = range(K_MIN, K_MAX + 1)

    for k in k_range:
        km     = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        wcss   = km.inertia_
        sil    = silhouette_score(X, labels, sample_size=min(500, len(X)), random_state=42)
        wcss_list.append(wcss)
        sil_list.append(sil)
        print(f"{k:>4} | {wcss:>14,.0f} | {sil:>10.4f}")

    best_idx  = int(np.argmax(sil_list))
    optimal_k = K_MIN + best_idx
    print(f"\n🎯 K tối ưu = {optimal_k} (Silhouette={sil_list[best_idx]:.4f})")
    print("   Căn cứ: Rousseeuw (1987) — Silhouette Score cao nhất")

    _plot_evaluation(list(k_range), wcss_list, sil_list, optimal_k)
    return optimal_k


def _plot_evaluation(k_range, wcss_list, sil_list, optimal_k):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, wcss_list, "bo-", linewidth=2, markersize=6)
    ax1.axvline(x=optimal_k, color="r", linestyle="--", label=f"K tối ưu = {optimal_k}")
    ax1.set_title("Elbow Method (WCSS)\nMacQueen (1967)", fontsize=12)
    ax1.set_xlabel("Số cụm Persona (K)")
    ax1.set_ylabel("WCSS")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_range, sil_list, "rs-", linewidth=2, markersize=6)
    ax2.axvline(x=optimal_k, color="r", linestyle="--",
                label=f"K tối ưu = {optimal_k} (Score={max(sil_list):.4f})")
    ax2.set_title("Silhouette Score\nRousseeuw (1987)", fontsize=12)
    ax2.set_xlabel("Số cụm Persona (K)")
    ax2.set_ylabel("Score (cao hơn = tốt hơn)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Xác định số lượng Persona tối ưu từ Embedding bài báo", fontsize=13)
    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", dpi=300, bbox_inches="tight")
    print("📊 Đã lưu biểu đồ: elbow_silhouette.png")
    plt.show()


# ─────────────────────────────────────────────
# STEP 3: CLUSTERING
# ─────────────────────────────────────────────
def run_clustering(X: np.ndarray, k: int):
    print(f"\n🔄 Chạy K-Means với K={k}...")
    km     = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    dist   = {i: int((labels == i).sum()) for i in range(k)}
    print(f"✅ Phân bố cụm: {dist}")
    return labels, km


# ─────────────────────────────────────────────
# STEP 4: ĐẶT TÊN PERSONA BẰNG OLLAMA
# ─────────────────────────────────────────────
def name_persona(cluster_id: int, sample_titles: list, sample_topics: list) -> dict:
    titles_str = "\n".join(
        f"- {t} [{top}]" for t, top in zip(sample_titles, sample_topics)
    )
    user_msg = f"""Dưới đây là {len(sample_titles)} bài báo đại diện cho một nhóm độc giả:

{titles_str}

Dựa vào nội dung các bài báo này, hãy mô tả nhóm độc giả bằng JSON:
{{
  "name": "Tên persona ngắn gọn 3-5 từ tiếng Việt",
  "icon": "1 emoji đại diện",
  "desc": "Mô tả 1 câu ngắn về sở thích của nhóm này",
  "prompt": "System prompt cho chatbot phục vụ nhóm này, 2-3 câu tiếng Việt, bắt đầu bằng Bạn là..."
}}

Chỉ trả về JSON hợp lệ, không markdown, không giải thích thêm."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Bạn là chuyên gia phân tích hành vi người dùng. Chỉ trả về JSON hợp lệ, không markdown, không giải thích.",
            },
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        result["cluster_id"] = cluster_id
        return result
    except Exception as e:
        print(f"  ⚠️  Ollama lỗi cụm {cluster_id}: {e}")
        return {
            "cluster_id": cluster_id,
            "name":       f"Nhóm độc giả {cluster_id + 1}",
            "icon":       "📖",
            "desc":       "Nhóm độc giả tổng hợp",
            "prompt":     "Bạn là trợ lý tin tức. Hãy trả lời ngắn gọn và khách quan.",
        }


# ─────────────────────────────────────────────
# STEP 5: VISUALIZE PCA 2D
# ─────────────────────────────────────────────
def visualize_clusters(X: np.ndarray, labels: np.ndarray, personas: list):
    print("\n🎨 Vẽ biểu đồ phân cụm (PCA 2D)...")
    pca    = PCA(n_components=2, random_state=42)
    X2d    = pca.fit_transform(X)
    colors = plt.cm.Set2(np.linspace(0, 1, len(personas)))

    plt.figure(figsize=(10, 7))
    for p, color in zip(personas, colors):
        cid  = p["cluster_id"]
        mask = labels == cid
        plt.scatter(
            X2d[mask, 0], X2d[mask, 1],
            c=[color],
            label=f"{p['icon']} {p['name']} ({mask.sum()} bài)",
            alpha=0.6, s=40,
        )

    plt.title("Phân cụm Persona từ Embedding bài báo (PCA 2D)", fontsize=13)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("persona_clusters.png", dpi=300, bbox_inches="tight")
    print("📊 Đã lưu biểu đồ: persona_clusters.png")
    plt.show()


# ─────────────────────────────────────────────
# STEP 6: LƯU PERSONA_CONFIG.JSON
# ─────────────────────────────────────────────
def save_persona_config(personas: list, optimal_k: int, total_articles: int):
    persona_config = {}
    for p in personas:
        persona_config[p["name"]] = {
            "icon":       p.get("icon", "📖"),
            "desc":       p["desc"],
            "prompt":     p["prompt"],
            "cluster_id": p["cluster_id"],
            "topics":     [],
        }

    output = {
        "metadata": {
            "optimal_k":      optimal_k,
            "total_articles": total_articles,
            "method":         "KMeans + Silhouette Score",
            "reference":      "MacQueen (1967); Rousseeuw (1987)",
        },
        "persona_config": persona_config,
    }

    with open("persona_config.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n📄 Đã lưu: persona_config.json")
    print("   → Copy PERSONA_CONFIG từ file này vào app.py")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  PERSONA CLUSTERING")
    print("=" * 50)

    # 1. Load data
    ids, titles, topics, X = fetch_embeddings()
    if len(X) == 0:
        print("❌ Không có dữ liệu, kiểm tra lại database.")
        return

    # 2. Tìm k tối ưu
    optimal_k = find_optimal_k(X)

    override = input(f"\nNhấn Enter để dùng K={optimal_k}, hoặc nhập số khác: ").strip()
    if override.isdigit() and int(override) >= K_MIN:
        optimal_k = int(override)
        print(f"→ Dùng K={optimal_k} theo lựa chọn thủ công")

    # 3. Clustering
    labels, km = run_clustering(X, optimal_k)

    # 4. Đặt tên persona
    print(f"\n🤖 Dùng Ollama đặt tên {optimal_k} persona...")
    personas = []
    for cid in range(optimal_k):
        mask    = labels == cid
        indices = np.where(mask)[0]

        # Lấy TOP_N bài gần centroid nhất
        centroid  = km.cluster_centers_[cid]
        distances = np.linalg.norm(X[indices] - centroid, axis=1)
        top_idx   = indices[np.argsort(distances)[:TOP_N]]

        sample_titles = [titles[i] for i in top_idx]
        sample_topics = [topics[i] for i in top_idx]

        print(f"\n  Cụm {cid+1}/{optimal_k} ({mask.sum()} bài):")
        for t in sample_titles:
            print(f"    • {t[:80]}")

        persona = name_persona(cid, sample_titles, sample_topics)
        print(f"  → {persona['icon']} {persona['name']}")
        personas.append(persona)

    # 5. Visualize
    visualize_clusters(X, labels, personas)

    # 6. Lưu kết quả
    save_persona_config(personas, optimal_k, len(X))

    # Tóm tắt
    print("\n" + "=" * 50)
    print(f"✅ HOÀN THÀNH! Số Persona tối ưu: {optimal_k}")
    print("=" * 50)
    for p in personas:
        print(f"\n  {p['icon']} {p['name']}")
        print(f"     {p['desc']}")
        print(f"     Prompt: {p['prompt'][:80]}...")
    print("\n→ Mở persona_config.json và copy vào PERSONA_CONFIG trong app.py")


if __name__ == "__main__":
    main()
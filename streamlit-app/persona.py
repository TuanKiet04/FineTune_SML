import psycopg2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB",   "optimize"),
    "user":     os.getenv("PG_USER", "kietcorn"),
    "password": os.getenv("PG_PASS", "kiietqo9204"),
}


def fetch_embeddings():
    print("⏳ Đang kết nối Database và tải dữ liệu...")
    conn = psycopg2.connect(**DB_CONFIG)
    vectors = []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT embedding::text 
            FROM n8n_vectors 
            WHERE embedding IS NOT NULL
        """)
        for row in cur.fetchall():
            vectors.append(json.loads(row[0]))
    conn.close()
    X = np.array(vectors, dtype=np.float32)
    print(f"📊 Đã tải {len(X)} vectors, {X.shape[1]} chiều.")
    return X

def evaluate_kmeans(X, k_min=2, k_max=10):
    wcss, silhouettes = [], []
    print(f"\n🧠 Chạy K-Means từ K={k_min} đến K={k_max}...")
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        sil = silhouette_score(X, labels, sample_size=min(500, len(X)), random_state=42)
        silhouettes.append(sil)
        print(f"  K={k:2d} | WCSS={km.inertia_:,.0f} | Silhouette={sil:.4f}")
    return wcss, silhouettes

def find_optimal_k(silhouettes, k_min):
    best_idx = int(np.argmax(silhouettes))
    return k_min + best_idx

def plot_results(wcss, silhouettes, k_min, k_max, optimal_k):
    k_range = range(k_min, k_max + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow
    ax1.plot(list(k_range), wcss, "bo-", linewidth=2, markersize=6)
    ax1.axvline(x=optimal_k, color='r', linestyle='--',
                label=f"K tối ưu = {optimal_k}")
    ax1.set_title("Elbow Method (WCSS)", fontsize=13)
    ax1.set_xlabel("Số cụm Persona (K)")
    ax1.set_ylabel("WCSS")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Silhouette
    ax2.plot(list(k_range), silhouettes, "rs-", linewidth=2, markersize=6)
    ax2.axvline(x=optimal_k, color='r', linestyle='--',
                label=f"K tối ưu = {optimal_k} (Silhouette={max(silhouettes):.4f})")
    ax2.set_title("Silhouette Score\n(Rousseeuw, 1987)", fontsize=13)
    ax2.set_xlabel("Số cụm Persona (K)")
    ax2.set_ylabel("Score (cao hơn = tốt hơn)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Xác định số lượng Persona tối ưu từ Embedding bài báo", fontsize=14)
    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", dpi=300, bbox_inches="tight")
    print(f"\n✅ Đã lưu biểu đồ: elbow_silhouette.png")
    plt.show()

if __name__ == "__main__":
    X = fetch_embeddings()
    K_MIN, K_MAX = 2, 10
    wcss, silhouettes = evaluate_kmeans(X, K_MIN, K_MAX)
    optimal_k = find_optimal_k(silhouettes, K_MIN)
    print(f"\n🎯 Số Persona tối ưu: K={optimal_k} (Silhouette={max(silhouettes):.4f})")
    print("   Căn cứ: Rousseeuw (1987) - Silhouette Score cao nhất")
    plot_results(wcss, silhouettes, K_MIN, K_MAX, optimal_k)
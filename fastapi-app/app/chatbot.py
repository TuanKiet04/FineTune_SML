from sqlalchemy.orm import Session
from sqlalchemy import text
import os
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def get_chat_response(question: str, db: Session, context_limit: int = 5):
    """
    Chatbot sử dụng PGVector để tìm bài báo liên quan
    và Ollama để generate câu trả lời
    """
    
    # 1. Tìm các bài báo liên quan bằng semantic search
    # Giả sử bạn đã có embeddings trong bảng (từ n8n workflow)
    # Query này tìm bài gần nhất với question (cần có embedding của question)
    
    # Đơn giản hóa: lấy bài mới nhất có chứa từ khóa
    keywords = question.lower().split()
    relevant_articles = []
    
    # Simple keyword search (trong production nên dùng pgvector proper)
    for keyword in keywords[:3]:  # Lấy 3 từ đầu tiên
        articles = db.execute(
            text("""
                SELECT title, content, url 
                FROM raw_data 
                WHERE LOWER(title) LIKE :keyword OR LOWER(content) LIKE :keyword
                ORDER BY published_at DESC 
                LIMIT :limit
            """),
            {"keyword": f"%{keyword}%", "limit": context_limit}
        ).fetchall()
        
        relevant_articles.extend(articles)
    
    # Remove duplicates
    seen = set()
    unique_articles = []
    for article in relevant_articles:
        if article[2] not in seen:  # url
            seen.add(article[2])
            unique_articles.append(article)
    
    # 2. Tạo context từ các bài báo
    context = "\n\n".join([
        f"Bài báo: {article[0]}\nNội dung: {article[1][:500]}..."
        for article in unique_articles[:context_limit]
    ])
    
    # 3. Gọi Ollama để generate answer
    prompt = f"""Dựa vào các bài báo sau, hãy trả lời câu hỏi của người dùng một cách chính xác và ngắn gọn.

Các bài báo liên quan:
{context}

Câu hỏi: {question}

Trả lời (bằng tiếng Việt):"""

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": "phi3:mini",  # Hoặc model bạn đã pull trong Ollama
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                answer = response.json().get("response", "Xin lỗi, tôi không thể trả lời câu hỏi này.")
            else:
                answer = "Lỗi khi gọi Ollama API"
    except Exception as e:
        answer = f"Lỗi: {str(e)}"
    
    return {
        "answer": answer,
        "sources": [
            {"title": article[0], "url": article[2]}
            for article in unique_articles[:3]
        ]
    }
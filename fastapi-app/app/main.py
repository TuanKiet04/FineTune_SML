# fastapi-app/app/main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
import os

from app.database import get_db, engine
from app import models, schemas, auth
from app.chatbot import get_chat_response

# Tạo tables nếu chưa có
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="News Chatbot API",
    description="API để đọc tin tức và chat với AI về các bài báo",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def index():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

# ============================================
# HEALTH CHECK
# ============================================
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fastapi-news"}


# ============================================
# AUTHENTICATION ROUTES
# ============================================
@app.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Đăng ký tài khoản mới"""
    # Check user tồn tại
    db_user = db.query(models.User).filter(
        (models.User.username == user.username) | 
        (models.User.email == user.email)
    ).first()
    
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Username hoặc email đã tồn tại"
        )
    
    # Tạo user mới
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        password_hash=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user


@app.post("/token", response_model=schemas.Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login và nhận JWT token"""
    user = db.query(models.User).filter(
        models.User.username == form_data.username
    ).first()
    
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username hoặc password không đúng",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/me", response_model=schemas.UserResponse)
def get_current_user_info(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Lấy thông tin user hiện tại"""
    user = auth.get_current_user(token, db)
    return user


# ============================================
# NEWS ROUTES
# ============================================
@app.get("/news", response_model=List[schemas.NewsArticle])
def get_news(
    skip: int = 0,
    limit: int = 20,
    topic: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Lấy danh sách tin tức
    - skip: bỏ qua n bài đầu (pagination)
    - limit: số lượng bài trả về
    - topic: lọc theo chủ đề (Cong Nghe, Thoi Su, ...)
    - search: tìm kiếm trong title
    """
    query = db.query(models.RawData).order_by(models.RawData.published_at.desc())
    
    if topic:
        query = query.filter(models.RawData.topic == topic)
    
    if search:
        query = query.filter(models.RawData.title.contains(search))
    
    articles = query.offset(skip).limit(limit).all()
    return articles


@app.get("/news/{article_id}", response_model=schemas.NewsArticle)
def get_news_detail(article_id: int, db: Session = Depends(get_db)):
    """Lấy chi tiết 1 bài báo"""
    article = db.query(models.RawData).filter(models.RawData.id == article_id).first()
    
    if not article:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài báo")
    
    return article


@app.get("/topics")
def get_topics(db: Session = Depends(get_db)):
    """Lấy danh sách các chủ đề có sẵn"""
    topics = db.query(models.RawData.topic).distinct().all()
    return {"topics": [t[0] for t in topics if t[0]]}


# ============================================
# CHAT ROUTES
# ============================================
@app.post("/chat", response_model=schemas.ChatResponse)
def chat(
    request: schemas.ChatRequest,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Chat với AI về tin tức
    - question: câu hỏi của user
    - context_limit: số lượng bài báo liên quan để tham khảo
    """
    user = auth.get_current_user(token, db)
    
    # Gọi chatbot
    response = get_chat_response(
        question=request.question,
        db=db,
        context_limit=request.context_limit
    )
    
    # Lưu lịch sử chat
    chat_history = models.ChatHistory(
        user_id=user.id,
        message=request.question,
        response=response["answer"]
    )
    db.add(chat_history)
    db.commit()
    
    return response


@app.get("/chat/history", response_model=List[schemas.ChatHistoryResponse])
def get_chat_history(
    limit: int = 50,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Lấy lịch sử chat của user"""
    user = auth.get_current_user(token, db)
    
    history = db.query(models.ChatHistory).filter(
        models.ChatHistory.user_id == user.id
    ).order_by(
        models.ChatHistory.created_at.desc()
    ).limit(limit).all()
    
    return history


# ============================================
# STATISTICS (BONUS)
# ============================================
@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Thống kê tổng quan"""
    total_articles = db.query(models.RawData).count()
    total_users = db.query(models.User).count()
    
    # Đếm bài theo topic
    from sqlalchemy import func
    topics_count = db.query(
        models.RawData.topic,
        func.count(models.RawData.id)
    ).group_by(models.RawData.topic).all()
    
    return {
        "total_articles": total_articles,
        "total_users": total_users,
        "articles_by_topic": {topic: count for topic, count in topics_count}
    }
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class NewsArticle(BaseModel):
    id: int
    url: str
    title: str
    content: Optional[str] = None
    topic: Optional[str] = None
    published_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    question: str
    context_limit: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []

class ChatHistoryResponse(BaseModel):
    id: int
    message: str
    response: str
    created_at: datetime
    
    class Config:
        from_attributes = True
from fastapi import FastAPI
from pydantic import BaseModel

from app.data import Documents
from app.embeddings import get_embedding, get_embeddings, using_real_model
from app.similarity import top_k_similar

app = FastAPI(title="Travel policy semantic search")
_DOC_TEXTS = [ doc["text"] for doc in Documents]
_DOC_VECTORS = get_embedding(_DOC_TEXTS)

class SearchRequest(BaseModel):
    query:str
    top_k:int=3
    min_score: float =0.0
    
class SearchResult(BaseModel):
    txt:str
    score:float

class SearchResponse(BaseModel):
    query:str
    results:list[SearchResult]
    
@app.get("/health")
def health():
    return {
        "status":"OK",
        "model":using_real_model(),
        "documents_indexed":len(Documents)
    }

@app.post("/search", response_model=SearchResponse)
def search(request:SearchRequest):
    query_vector = get_embedding(request.query)
    ranked = top_k_similar(query_vector,_DOC_VECTORS, k=request.top_k)
    results = [
        SearchResult(txt=_DOC_TEXTS[idx], score=round(score, 4))
        for idx, score in ranked
        if score >= request.min_score
    ]
    return SearchResponse(query=request.query, results=results)
"""
similarity.py
-------------
Pure math. No FastAPI, no models — this file should be understandable
and testable in complete isolation. That's intentional: in interviews,
being able to write cosine similarity from scratch on a whiteboard is
a very common check.
"""
import numpy as np


def cosine_similarity(vector_a, vector_b)-> float:
    a= np.array(vector_a,dtype=np.float32)
    b= np.array(vector_b, dtype=np.float32)
    
    dot= np.dot(a,b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    
    if mag_a ==0 or mag_b==0:
        return 0
    
    return float(dot/(mag_a*mag_b))


def top_k_similar(query_request,doc,k:int=3)-> float:
    scores =[ cosine_similarity (query_request, d) for d in doc]
    rank = sorted(enumerate(scores), key = lambda pair: pair[1],reverse=True)
    return rank[:k]
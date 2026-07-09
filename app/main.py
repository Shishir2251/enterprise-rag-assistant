from fastapi import FastAPI

app = FastAPI(title="Enterprise RAG Assistant")

@app.get("/")
def health_check():
    return {"message": "Enterprise RAG Assistant is running"}
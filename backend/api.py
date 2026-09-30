from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent


app = FastAPI(
    title="HYVE API",
    description="Backend API for the HYVE Hybrid RAG system",
    version="1.0.0"
)


# Allow the React development server to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "name": "HYVE",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/query")
def query_rag(request: QueryRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        answer = agent.ask_assistant(question)

        results = agent.hybrid_search(
            question,
            top_k=4
        )

        sources = []

        for result in results:
            sources.append({
                "id": result["id"],
                "title": result["title"],
                "text": result["text"],
                "vector_score": round(
                    float(result.get("vector_score", 0)),
                    3
                ),
                "keyword_score": round(
                    float(result.get("keyword_score", 0)),
                    3
                ),
                "hybrid_score": round(
                    float(result.get("hybrid_score", 0)),
                    3
                )
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

class EvaluationRequest(BaseModel):
    session: list


@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    if not request.session:
        raise HTTPException(
            status_code=400,
            detail="No questions in the current session."
        )

    try:
        evaluation = agent.evaluate_session(request.session)

        return {
            "evaluation": evaluation,
            "questions_evaluated": len(request.session)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt",
        ".md"
    }

    filename = file.filename.lower()

    if not any(
        filename.endswith(extension)
        for extension in allowed_extensions
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, TXT and MD files are supported."
        )

    try:

        contents = await file.read()

        result = agent.process_uploaded_document(
            contents,
            file.filename
        )

        return {
            "message": "Document uploaded successfully.",
            "filename": file.filename,
            "documents": result["documents"],
            "chunks": result["chunks"],
            "characters": result["characters"]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
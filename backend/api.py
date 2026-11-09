from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
import time
from collections import defaultdict
from pydantic import BaseModel, Field
import json
from fastapi.responses import StreamingResponse


import agent


app = FastAPI(
    title="HYVE API",
    description="Backend API for the HYVE Hybrid RAG system",
    version="1.0.0"
)

RATE_LIMITS = {
    "/": (60, 60),
    "/health": (60, 60),
    "/query": (10, 60),
    "/query/stream": (10, 60),
    "/evaluate": (5, 60),
    "/upload": (5, 600),
}

request_history = defaultdict(list)


def check_rate_limit(request: Request):
    path = request.url.path

    if path not in RATE_LIMITS:
        return

    max_requests, window = RATE_LIMITS[path]

    # Use client IP as the limiter key
    client_ip = request.client.host if request.client else "unknown"
    key = f"{client_ip}:{path}"

    now = time.time()

    # Keep only requests inside the current window
    request_history[key] = [
        timestamp
        for timestamp in request_history[key]
        if now - timestamp < window
    ]

    if len(request_history[key]) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later.",
            headers={"Retry-After": str(window)}
        )

    request_history[key].append(now)


# Allow the React development server to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000
    )


@app.get("/")
def root(request: Request):
    check_rate_limit(request)
    return {
        "name": "HYVE",
        "status": "online"
    }


@app.get("/health")
def health(request: Request):
    check_rate_limit(request)
    return {
        "status": "healthy"
    }

@app.post("/query/stream")
def query_stream(request: Request, query: QueryRequest):

    check_rate_limit(request)

    question = query.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    def generate():

        try:
            # Retrieve sources for the frontend
            results = agent.hybrid_search(
                question,
                top_k=4
            )

            sources = []

            for result in results:
                sources.append({
                    "chunk_id": result["id"],
                    "title": result["title"],
                    "text": result["text"],
                    "vector_score": round(
                        float(result.get("vector_score", 0)), 3
                    ),
                    "keyword_score": round(
                        float(result.get("keyword_score", 0)), 3
                    ),
                    "hybrid_score": round(
                        float(result.get("hybrid_score", 0)), 3
                    )
                })

            # Send retrieved sources first
            yield json.dumps({
                "type": "sources",
                "sources": sources
            }) + "\n"

            # Stream the generated answer
            for token in agent.stream_assistant(question):

                yield json.dumps({
                    "type": "token",
                    "text": token
                }) + "\n"

            # Tell the frontend generation is complete
            yield json.dumps({
                "type": "done"
            }) + "\n"

        except Exception:
            yield json.dumps({
                "type": "error",
                "message": "Unable to generate the answer."
            }) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )

@app.post("/query")
def query_rag(request: Request, query: QueryRequest):
    check_rate_limit(request)

    question = query.question.strip()

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
        print("Query processing failed.")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the query."
        )

class EvaluationRequest(BaseModel):
    session: list = Field(
        ...,
        min_length=1,
        max_length=50
    )


@app.post("/evaluate")
def evaluate(request: Request, eval_request: EvaluationRequest):
    check_rate_limit(request)

    if not eval_request.session:
        raise HTTPException(
            status_code=400,
            detail="No questions in the current session."
        )

    try:
        evaluation = agent.evaluate_session(eval_request.session)

        return {
            "evaluation": evaluation,
            "questions_evaluated": len(eval_request.session)
        }

    except Exception as error:
        print("Evaluation failed.")

        raise HTTPException(
            status_code=500,
            detail="Unable to evaluate the session."
        )

@app.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...)
):
    check_rate_limit(request)

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
        
        MAX_FILE_SIZE = 25 * 1024 * 1024

        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum size is 25 MB."
            )

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
    except HTTPException:
        raise

    except Exception as error:
        print("Upload failed.")

        raise HTTPException(
            status_code=500,
            detail="Unable to upload the document."
        )
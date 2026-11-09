import React, { useState } from "react";
import {
  MessageSquare, BarChart3,
  Trash2, Send, CircleCheck, Database
} from "lucide-react";
import "./style.css";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const NAV = [
  ["chat", "Chat", MessageSquare],
  ["evaluation", "Evaluation", BarChart3]
];

const API_BASE_URL =
  window.huggingface?.variables?.API_URL ||
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

function App() {
  const [page, setPage] = useState("chat");

  //chat state
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [sources, setSources] = useState([]);
  const [generating, setGenerating] = useState(false);

  //Evaluation State
  const [sessionResults, setSessionResults] = useState([]);
  const [evaluation, setEvaluation] = useState(null);
  const [evaluating, setEvaluating] = useState(false);
  const [knowledgeBase, setKnowledgeBase] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  const ask = async (value = query) => {
      const text = value.trim();
      if (!text || generating) return;

      setGenerating(true);
      setQuery("");

      // Add user question and an empty assistant message
      setMessages(current => [
          ...current,
          { role: "user", text },
          { role: "assistant", text: "" }
      ]);

      let answerText = "";
      let retrievedSources = [];

      try {
          const response = await fetch(
              `${API_BASE_URL}/query/stream`,
              {
                  method: "POST",
                  headers: {
                      "Content-Type": "application/json"
                  },
                  body: JSON.stringify({
                      question: text
                  })
              }
          );

          if (!response.ok) {
              throw new Error("Backend request failed.");
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();

          let buffer = "";

          const processEvent = (line) => {
              if (!line.trim()) return;

              const event = JSON.parse(line);

              if (event.type === "sources") {
                  retrievedSources = event.sources || [];
                  setSources(retrievedSources);
              }

              else if (event.type === "token") {
                  answerText += event.text || "";

                  setMessages(current => {
                      const updated = [...current];

                      updated[updated.length - 1] = {
                          ...updated[updated.length - 1],
                          text: answerText
                      };

                      return updated;
                  });
              }

              else if (event.type === "error") {
                  throw new Error(
                      event.message || "Answer generation failed."
                  );
              }
          };

          // Read incoming data progressively
          while (true) {
              const { value, done } = await reader.read();

              if (done) break;

              buffer += decoder.decode(value, {
                  stream: true
              });

              const lines = buffer.split("\n");
              buffer = lines.pop();

              for (const line of lines) {
                  processEvent(line);
              }
          }

          // Process remaining buffered data
          buffer += decoder.decode();

          if (buffer.trim()) {
              processEvent(buffer);
          }

          // Save completed interaction for evaluation
          setSessionResults(current => [
              ...current,
              {
                  Question: text,
                  Context: retrievedSources
                      .map(source => source.text)
                      .join("\n"),
                  Answer: answerText
              }
          ]);

      } catch (error) {
          console.error("Streaming error:", error);

          setMessages(current => {
              const updated = [...current];

              updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  text: answerText
                      ? answerText + "\n\nUnable to complete the response."
                      : "Unable to connect to the RAG backend."
              };

              return updated;
          });

      } finally {
          setGenerating(false);
      }
  };

  //RUN SESSION EVALUATION

  const runEvaluation = async () => {
    if (sessionResults.length === 0) {
      alert("Ask at least one question before running evaluation.");
      return;
    }

    setEvaluating(true);

    try {
      const response = await fetch(`${API_BASE_URL}/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          session: sessionResults
        })
      });

      if (!response.ok) {
        throw new Error("Evaluation request failed");
      }

      const data = await response.json();

      const scores = {};
      const lines = data.evaluation.split("\n");

      lines.forEach(line => {
        const match = line.match(/^(.+):\s*(\d+(?:\.\d+)?)\/5$/);

        if (match) {
          scores[match[1].trim()] = Number(match[2]);
        }
      });

      setEvaluation({
        scores,
        questions: data.questions_evaluated
      });

    } catch (error) {
      console.error(error);
      alert("Unable to run evaluation.");
    } finally {
      setEvaluating(false);
    }
    };

  return (
    <div className="shell">
      <Sidebar
        page={page}
        setPage={setPage}
        knowledgeBase={knowledgeBase}
        clear={() => {
            setMessages([]);
            setSources([]);
            setSessionResults([]);
            setEvaluation(null);
            setKnowledgeBase(null);
            setUploadError("");
        }}
      />

      <main className="main">
        {page === "chat" && (
          <Chat
            messages={messages}
            query={query}
            setQuery={setQuery}
            ask={ask}
            sources={sources}
            setMessages={setMessages}
            setSources={setSources}
            knowledgeBase={knowledgeBase}
            setKnowledgeBase={setKnowledgeBase}
            uploading={uploading}
            setUploading={setUploading}
            uploadError={uploadError}
            setUploadError={setUploadError}
            generating={generating}
          />
        )}

        {page === "evaluation" && (
          <Evaluation
            evaluation={evaluation}
            evaluating={evaluating}
            runEvaluation={runEvaluation}
            questionCount={sessionResults.length}
          />
        )}
      </main>
    </div>
  );
}

function Sidebar({page,setPage,knowledgeBase,clear}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">H</div>
        <div><strong>HYVE</strong><small>KNOWLEDGE ENGINE</small></div>
      </div>

      <div className="label">WORKSPACE</div>
      <nav>
        {NAV.map(([id,name,Icon]) => (
          <button className={`nav-item ${page===id ? "active":""}`} onClick={()=>setPage(id)} key={id}>
            <Icon size={17}/><span>{name}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <div className="label">SYSTEM</div>
        {["FAISS","BM25","Qwen 2.5 3B"].map((name,i)=>(
          <div className="status" key={name}>
            <CircleCheck size={14}/>{name}<em>{i===2 ? "READY":"ONLINE"}</em>
          </div>
        ))}

        <div className="stats">
          <div>
            <b>{knowledgeBase?.documents ?? 0}</b>            
            <small>documents</small>
          </div>

          <div>
            <b>{knowledgeBase?.chunks ?? 0}</b>
            <small>chunks</small>
          </div>
        </div>

        <button className="clear" onClick={clear}><Trash2 size={14}/> Clear conversation</button>
      </div>
    </aside>
  );
}

function Header({eyebrow,title,copy}) {
  return (
    <header>
      <div className="eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      <p className="copy">{copy}</p>
      <hr/>
    </header>
  );
}

function Chat({
  messages,
  query,
  setQuery,
  ask,
  sources,
  setMessages,
  setSources,
  knowledgeBase,
  setKnowledgeBase,
  uploading,
  setUploading,
  uploadError,
  setUploadError,
  generating
}) {  
  
    const uploadDocument = async (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setUploading(true);
    setUploadError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed.");
      }

      setKnowledgeBase({
        filename: data.filename,
        documents: data.documents,
        chunks: data.chunks,
        characters: data.characters
      });

      setMessages([]);
      setSources([]);
    } catch (error) {
      setUploadError(error.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };
  return (
    <>
      <Header eyebrow="CHAT INTERFACE / 01"
        title={<>Ask HYVE.<br/><span>Get answers you can trace.</span></>}
        copy="Upload a document, then ask questions and trace HYVE's answers back to the retrieved sources."      />

      <div className="chat-grid">
        <section>
          <div className="section-title">CONVERSATION</div>
            <div className="upload-bar">
              <label className="upload-button">
                <Database size={16} />
                {uploading ? "UPLOADING..." : "UPLOAD DOCUMENT"}
                <input
                  type="file"
                  accept=".pdf,.docx,.txt,.md"
                  onChange={uploadDocument}
                  disabled={uploading}
                  hidden
                />
              </label>

              {knowledgeBase && (
                <div className="upload-status">
                <strong>ACTIVE · {knowledgeBase.filename}</strong>
                <span>
                  {knowledgeBase.documents} document · {knowledgeBase.chunks} chunks ·{" "}
                  {knowledgeBase.characters.toLocaleString()} characters
                </span>
              </div>
              )}
            </div>

            {uploadError && (
              <div className="upload-error">
                {uploadError}
              </div>
            )}

          {!messages.length ? (
            <div className="empty">
              {knowledgeBase ? (
                <>
                  <strong>Document ready.</strong>
                  <p>
                    Ask HYVE questions about <b>{knowledgeBase.filename}</b>.
                  </p>

                  <div className="quick-hints">
                    <span>SUMMARIZE</span>
                    <span>EXPLAIN A SECTION</span>
                    <span>FIND INFORMATION</span>
                  </div>
                </>
              ) : (
                <>
                  <strong>Upload a document to get started.</strong>
                  <p>
                    Upload a PDF, DOCX, TXT, or MD file, then ask questions about its content.
                  </p>
                </>
              )}
            </div>
          ) : (
            <div className="messages">
              {messages.map((m, i) => (
                <div className={`message ${m.role}`} key={i}>
                  <b>{m.role === "user" ? "YOU" : "H"}</b>

                  <div>
                    <small>
                      {m.role === "user" ? "QUESTION" : "HYVE"}
                    </small>

                    {m.role === "assistant" && !m.text && generating ? (
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    ) : m.role === "assistant" ? (
                      <div className="markdown-content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {m.text}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p>{m.text}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
        )}

          <div className="input">
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === "Enter" && ask()}
              placeholder={
                knowledgeBase
                  ? `Ask about ${knowledgeBase.filename}...`
                  : "Upload a document to start asking questions..."
              }
              disabled={!knowledgeBase || uploading || generating}
            />

            <button
              onClick={() => ask()}
              disabled={!knowledgeBase || uploading || generating}
            >
              <Send size={17} />
            </button>
          </div>
        </section>

        <section>
                  {sources.length === 0 ? (
          <div className="evidence">
            <p>No retrieval results yet.</p>
          </div>
        ) : (
          sources.map((source) => (
            <div className="evidence" key={source.chunk_id}>
              <div className="evidence-top">
                <strong>{source.title}</strong>
                <span>{source.hybrid_score} HYBRID</span>
              </div>

              <small>{source.chunk_id}</small>

              <p>{source.text}</p>

              <div className="score">
                <i
                  style={{
                    width: `${Math.min(source.hybrid_score * 100, 100)}%`
                  }}
                />
              </div>
            </div>
          ))
        )}
        </section>
      </div>
    </>
  );
}

function Evaluation({
  evaluation,
  evaluating,
  runEvaluation,
  questionCount
}) {
  const scores = evaluation?.scores || {};

  const metrics = [
    ["Retrieval Relevance", scores["Retrieval Relevance"]],
    ["Answer Correctness", scores["Answer Correctness"]],
    ["Groundedness", scores["Groundedness"]],
    ["Completeness", scores["Completeness"]],
    ["Overall", scores["Overall"]]
  ];

  return (
    <>
      <Header
        eyebrow="EVALUATION / 05"
        title={
          <>
            Evaluate the <span>session.</span>
          </>
        }
        copy="Measure the quality of the questions and answers generated during the current conversation."
      />

      <div className="evaluation-hero">

        <div>
          <div className="section-title">
            SESSION RESULT
          </div>

          <strong>
            {evaluation &&
            scores["Overall"] !== undefined
              ? `${scores["Overall"].toFixed(2)}/5`
              : "—"}
          </strong>

          <p>
            {evaluation
              ? `Evaluation completed across ${evaluation.questions} questions.`
              : questionCount > 0
                ? `${questionCount} question${
                    questionCount === 1 ? "" : "s"
                  } ready for evaluation.`
                : "Ask questions in Chat, then run the evaluation."}
          </p>
        </div>

        <div className="method">

          <button
            className="run-evaluation"
            onClick={runEvaluation}
            disabled={
              evaluating || questionCount === 0
            }
          >
            {evaluating
              ? "RUNNING..."
              : "RUN EVALUATION"}
          </button>

          <small>SESSION</small>

          <b>
            {questionCount} question
            {questionCount === 1 ? "" : "s"}
          </b>

          <small>
            LLM-based 1–5 scoring
          </small>

        </div>

      </div>

      <div className="metrics">

        <div className="metrics-head">
          <span>METRIC</span>
          <span>SCORE</span>
        </div>

        {metrics.map(([name, score]) => (
          <div
            className="metrics-row"
            key={name}
          >
            <b>{name}</b>

            <span className="acid">
              {score !== undefined
                ? `${score.toFixed(2)}/5`
                : "—"}
            </span>
          </div>
        ))}

      </div>
    </>
  );
}

export default App;
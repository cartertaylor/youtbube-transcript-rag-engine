import logging
import os
from typing import Optional

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi


# Silence benign SDK warning when Gemini 2.5 Flash returns thought_signature parts
class _SuppressNonTextWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "there are non-text parts in the response" not in record.getMessage()


logging.getLogger("google_genai.types").addFilter(_SuppressNonTextWarning())

load_dotenv()

ai = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
chroma = chromadb.PersistentClient(path="./yt_chroma_db")
collection = chroma.get_or_create_collection(
    name="youtube_captions",
    metadata={"hnsw:space": "cosine"},
    embedding_function=None,
)


# --- 1. Pydantic Schemas (Structured Data & Validation) ---


class Citation(BaseModel):
    url: str = Field(description="Timestamped YouTube URL from the retrieved excerpts")
    quote: str = Field(description="Short verbatim phrase supporting the claim")


class RAGResponse(BaseModel):
    is_answerable: bool = Field(
        description="True only if the transcript excerpts explicitly answer the question"
    )
    answer: str = Field(
        description="Grounded answer synthesized strictly from retrieved excerpts"
    )
    citations: list[Citation] = Field(default_factory=list)


# --- 2. Chunking & Vector Ingestion ---


def chunk_transcript(
    video_id: str, window_sec: float = 45.0, overlap_sec: float = 10.0
) -> list[dict]:
    """Merge short caption lines into overlapping time windows with timestamp metadata."""
    raw_snippets = YouTubeTranscriptApi().fetch(video_id).to_raw_data()
    chunks = []
    i = 0

    while i < len(raw_snippets):
        start_time = raw_snippets[i]["start"]
        texts = []
        j = i

        while j < len(raw_snippets) and (raw_snippets[j]["start"] - start_time) < window_sec:
            texts.append(raw_snippets[j]["text"].replace("\n", " "))
            j += 1

        end_time = raw_snippets[j - 1]["start"] + raw_snippets[j - 1]["duration"]
        start_int = int(start_time)
        chunks.append({
            "id": f"{video_id}_{start_int}",
            "text": " ".join(texts),
            "metadata": {
                "video_id": video_id,
                "start_sec": start_int,
                "end_sec": int(end_time),
                "url": f"https://youtu.be/{video_id}?t={start_int}",
            },
        })

        next_start = end_time - overlap_sec
        prev_i = i
        while i < j and raw_snippets[i]["start"] < next_start:
            i += 1
        if i == prev_i:
            i = j

    return chunks


def ingest_video(video_id: str) -> None:
    """Chunk, embed, and upsert a video's transcript into ChromaDB."""
    chunks = chunk_transcript(video_id)
    if not chunks:
        raise ValueError(f"No caption chunks extracted for {video_id}")

    texts = [c["text"] for c in chunks]
    embed_res = ai.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    vectors = [e.values for e in embed_res.embeddings]

    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks for video {video_id}")


# --- 3. Agentic Retrieval Tool & Structured Generation ---


def search_captions(query: str, video_id: str = "", top_k: int = 5) -> str:
    """Search indexed YouTube transcripts semantically for relevant timestamped excerpts.

    Args:
        query: Search query phrased to match spoken concepts in the video.
        video_id: Optional YouTube video ID to restrict the search scope.
        top_k: Number of chunks to retrieve (1-8).
    """
    print(f"[Tool Call] search_captions(query={query!r}, video_id={video_id!r}, top_k={top_k})")
    try:
        q_embed = ai.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        ).embeddings[0].values

        where_filter: Optional[dict] = {"video_id": video_id} if video_id else None
        results = collection.query(
            query_embeddings=[q_embed],
            n_results=int(top_k),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            if dist <= 0.38:
                hits.append(f"[url={meta['url']} | cosine_dist={dist:.3f}]\n{doc}")

        return "\n\n".join(hits) if hits else "NO_RELEVANT_CHUNKS_FOUND"
    except Exception as exc:
        print(f"[Tool Error] {exc}")
        return f"SEARCH_ERROR: {exc}"


def agentic_ask(question: str) -> RAGResponse:
    """Run tool-calling retrieval followed by schema-validated JSON synthesis."""
    retrieved_evidence: list[str] = []

    def search_tool(query: str, video_id: str = "", top_k: int = 5) -> str:
        """Search indexed YouTube transcripts semantically for relevant timestamped excerpts.

        Args:
            query: Search query phrased to match spoken concepts in the video.
            video_id: Optional YouTube video ID to restrict the search scope.
            top_k: Number of chunks to retrieve (1-8).
        """
        raw_hits = search_captions(query=query, video_id=video_id, top_k=top_k)
        if raw_hits != "NO_RELEVANT_CHUNKS_FOUND" and not raw_hits.startswith("SEARCH_ERROR"):
            retrieved_evidence.append(raw_hits)
        return raw_hits

    agent_turn = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "You are a research assistant with access to a YouTube transcript vector database. "
            "Call `search_tool` to gather evidence for the user's question. Leave `video_id` "
            "empty unless the user specifies a video ID. If the first search returns "
            "NO_RELEVANT_CHUNKS_FOUND, retry once with broader keywords.\n\n"
            f"User Question: {question}"
        ),
        config=types.GenerateContentConfig(tools=[search_tool]),
    )

    if not retrieved_evidence:
        return RAGResponse(
            is_answerable=False,
            answer="No relevant segments were found in the indexed videos to answer this question.",
            citations=[],
        )

    raw_sources = "\n\n---\n\n".join(retrieved_evidence)
    draft_answer = agent_turn.text or ""

    structured_turn = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Build a grounded JSON response using ONLY the Raw Transcript Excerpts below. "
            "Every claim in `answer` must be supported by an entry in `citations` containing "
            "the exact `url` from the excerpt header and a short verbatim `quote` from that excerpt. "
            "Set `is_answerable=False` if the excerpts do not actually answer the question.\n\n"
            f"Draft Notes:\n{draft_answer}\n\n"
            f"Raw Transcript Excerpts:\n{raw_sources}\n\n"
            f"Question: {question}"
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RAGResponse,
        ),
    )
    return structured_turn.parsed

# --- 4. Retrieval Evaluation Harness ---


def run_diagnostics(eval_set: list[dict]) -> None:
    """Measure Recall@1, Recall@3, Recall@5, and MRR against target timestamps."""
    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    rr_scores = []

    for item in eval_set:
        q_vec = ai.models.embed_content(
            model="gemini-embedding-001",
            contents=item["q"],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        ).embeddings[0].values

        res = collection.query(
            query_embeddings=[q_vec],
            n_results=5,
            where={"video_id": item["video_id"]},
            include=["metadatas", "distances", "documents"],
        )

        metas = res["metadatas"][0]
        dists = res["distances"][0]
        docs = res["documents"][0]

        found_rank = None
        for idx, m in enumerate(metas, start=1):
            if m["start_sec"] <= item["target_sec"] <= m["end_sec"]:
                found_rank = idx
                break

        if found_rank == 1:
            hits_1 += 1
        if found_rank and found_rank <= 3:
            hits_3 += 1
        if found_rank and found_rank <= 5:
            hits_5 += 1
        rr_scores.append(1.0 / found_rank if found_rank else 0.0)

        status = f"HIT @ Rank #{found_rank}" if found_rank else "MISS (not in top 5)"
        print(f"\nQ: {item['q']} (target={item['target_sec']}s) -> {status}")
        for rank, (m, d, doc) in enumerate(zip(metas, dists, docs), start=1):
            preview = doc[:75].replace("\n", " ")
            print(f"  #{rank} [{m['start_sec']}s-{m['end_sec']}s] dist={d:.3f} | {preview}...")

    n = len(eval_set)
    mrr = sum(rr_scores) / n
    print(f"\nRecall@1: {hits_1}/{n} | Recall@3: {hits_3}/{n} | Recall@5: {hits_5}/{n} | MRR: {mrr:.2f}")


# --- 5. Entrypoint ---


if __name__ == "__main__":
    VIDEO_ID = "AuwN1O56Ti8"

    # Skip re-indexing if vectors are already persisted in ./yt_chroma_db
    if collection.count() == 0:
        ingest_video(VIDEO_ID)

    print("Ask questions about the video (type 'quit' or 'exit' to stop):")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if question:
            answer = agentic_ask(question)
            # print(answer.model_dump_json(indent=2))
            print ("------- Structured RAG Output ---")
            print(f"\nAnswer: {answer.answer}\n")


    #### Tests for accuracy and retrieval evaluation (uncomment to run) ####
    # # Test 1: Agentic RAG with validated Pydantic output
    # result = agentic_ask("What is a good weekly lifting volume?")
    # print("\n--- Structured RAG Output ---")
    # print(result.model_dump_json(indent=2))

    # # Test 2: Retrieval benchmark across your ground-truth timestamps
    # eval_questions = [
    #     {"q": "What is a good weekly lifting volume?", "video_id": VIDEO_ID, "target_sec": 120},
    #     # Add your remaining ground-truth questions and target timestamps here
    # ]
    # run_diagnostics(eval_questions)
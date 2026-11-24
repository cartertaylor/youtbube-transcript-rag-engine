# YouTube Transcript RAG

This project builds a local retrieval-augmented generation (RAG) system for a YouTube video. It downloads the video's transcript, splits it into overlapping timestamped chunks, creates Gemini embeddings for each chunk, and stores those embeddings in a local ChromaDB database. When you ask a question in the terminal, the program retrieves relevant transcript excerpts and asks Gemini to produce a structured, citation-backed answer.

## How it works

1. The YouTube transcript is fetched using the video ID.
2. Transcript lines are grouped into overlapping 45-second chunks.
3. Gemini creates an embedding for every chunk.
4. Chunks, embeddings, and timestamp URLs are saved to `./yt_chroma_db`.
5. Gemini embeds each question and searches ChromaDB for semantically similar chunks.
6. A grounded answer is generated in `RAGResponse` format, including timestamped citations.

Results remain local in `yt_chroma_db`, so indexing is only performed the first time the collection is empty.

## Requirements

- Python 3.9 or newer
- A Gemini API key
- Internet access for YouTube transcripts and Gemini API calls
- Python packages:

```bash
pip install youtube-transcript-api google-genai chromadb python-dotenv pydantic
```

## Gemini API setup

Create a file named `.env` in the project root (the same directory as `yt_rag.py`) and add:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Do not commit the `.env` file or API key to GitHub. The program loads the key with `load_dotenv()` and reads it from `GEMINI_API_KEY`.

## Changing the YouTube video

Find a YouTube video's ID from its URL:

```text
https://www.youtube.com/watch?v=AuwN1O56Ti8
                                └──── ID: AuwN1O56Ti8
```

Change `VIDEO_ID` near the bottom of `yt_rag.py`:

```python
if __name__ == "__main__":
    VIDEO_ID = "AuwN1O56Ti8"
```

Replace `AuwN1O56Ti8` with the ID of the video you want to index.

> If `yt_chroma_db` already contains indexed data, changing only `VIDEO_ID` is not enough. The entrypoint also skips ingestion whenever `collection.count() > 0`, even if the collection contains chunks for another video. To index a different video into a fresh database, delete or rename `./yt_chroma_db`, or update the ingestion logic to use a separate collection per video.

## Running the program

From the project directory, run:

```bash
python yt_rag.py
```

When prompted, enter a question. Type `quit` or `exit` to stop.

Example:

```text
Question: What is a good weekly lifting volume?
Answer: is_answerable=True answer='...' citations=[...]
```

## Function overview

### `_SuppressNonTextWarning`

A logging filter that suppresses a benign Gemini SDK warning generated when a response contains non-text parts such as thought-signature metadata.

### `Citation`

A Pydantic model representing one supporting citation. It stores:

- `url`: the timestamped YouTube URL for the excerpt.
- `quote`: a short verbatim quote from the transcript.

### `RAGResponse`

A Pydantic model defining Gemini's validated final response. It includes:

- `is_answerable`: whether the retrieved excerpts answer the question.
- `answer`: the grounded answer.
- `citations`: supporting timestamped citations.

This schema ensures Gemini returns structured data instead of unvalidated free-form text.

### `chunk_transcript(video_id, window_sec=45.0, overlap_sec=10.0)`

Fetches the video's transcript and splits it into overlapping windows.

- Default window: 45 seconds
- Default overlap: 10 seconds
- Returns chunks containing an ID, combined text, and metadata
- Metadata includes the video ID, start/end seconds, and timestamped YouTube URL

The overlap helps preserve context when a concept spans the boundary between adjacent chunks.

### `ingest_video(video_id)`

Indexes a video for retrieval:

1. Calls `chunk_transcript()`.
2. Generates Gemini embeddings using `gemini-embedding-001` and the `RETRIEVAL_DOCUMENT` task type.
3. Inserts or updates the vectors, transcript text, IDs, and metadata in ChromaDB.

If no chunks are produced, it raises a `ValueError`.

### `search_captions(query, video_id="", top_k=5)`

Performs semantic retrieval over indexed transcript chunks:

- Embeds the search query with the `RETRIEVAL_QUERY` task type.
- Optionally filters results to one video using `video_id`.
- Retrieves up to `top_k` chunks.
- Keeps only chunks with cosine distance below `0.38`.
- Formats matching chunks with their timestamped URLs and cosine distance.

It returns `NO_RELEVANT_CHUNKS_FOUND` when no chunk passes the threshold and `SEARCH_ERROR: ...` if retrieval fails.

### `agentic_ask(question)`

Runs the complete question-answer pipeline:

1. Gemini receives the question and can call the nested `search_tool` function.
2. The search tool forwards requests to `search_captions()` and records retrieved evidence.
3. If no evidence is found, the function returns a non-answerable response.
4. Gemini performs a second generation step using only the retrieved transcript excerpts.
5. Pydantic validates and parses the result as an `RAGResponse`.

If the first search finds nothing, Gemini is prompted to retry once using broader keywords.

### `run_diagnostics(eval_set)`

Measures retrieval quality against a list of known questions and target timestamps.

Each evaluation item requires:

```python
{
    "q": "Your evaluation question",
    "video_id": "AuwN1O56Ti8",
    "target_sec": 120,
}
```

The function reports:

- **Recall@1**: whether the correct timestamp is in the first result.
- **Recall@3**: whether the correct timestamp is in the top three results.
- **Recall@5**: whether the correct timestamp is in the top five results.
- **MRR**: mean reciprocal rank across evaluation questions.

## Optional evaluation tests

Retrieval tests are commented out in the entrypoint. To enable them, uncomment the test section and add ground-truth questions and target timestamps to `eval_questions`. You can also call:

```python
result = agentic_ask("What is a good weekly lifting volume?")
print(result.model_dump_json(indent=2))
```

## Project files

- `yt_rag.py`: transcript chunking, embeddings, retrieval, Gemini generation, and command-line entrypoint.
- `.env`: local Gemini API key configuration; this file should not be committed.
- `yt_chroma_db/`: local persistent ChromaDB vector database.
- `README.md`: project setup and function documentation.

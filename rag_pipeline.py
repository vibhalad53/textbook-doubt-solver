"""
rag_pipeline.py
================
The "brain" of the Textbook Doubt Solver. It contains every step of the
Retrieval-Augmented Generation (RAG) pipeline:

    PDF  ->  page text  ->  chunks  ->  embeddings  ->  FAISS index
    question -> top 3 chunks -> Gemini -> answer + verified quote + page

The seven functions required by the assignment are:
    1. extract_text_from_pdf()
    2. split_text_into_chunks()
    3. create_embeddings()
    4. build_faiss_index()
    5. retrieve_relevant_chunks()
    6. generate_answer()
    7. explain_simpler()

Heavy libraries (faiss, sentence-transformers, google-genai) are imported
inside the functions that need them. This keeps the file importable even
before everything is installed and gives clearer error messages.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from utils.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    NOT_FOUND_MESSAGE,
    SIMPLIFY_PROMPT_TEMPLATE,
    SIMPLIFY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # small, free, runs on CPU
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"  # Google's alias for the newest Flash model
FALLBACK_GEMINI_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
CHUNK_SIZE = 800        # characters per chunk (roughly 120-150 words)
CHUNK_OVERLAP = 150     # characters repeated between neighbouring chunks
TOP_K = 3               # number of chunks to retrieve for every question
MIN_TEXT_CHARS = 50     # less text than this = probably a scanned PDF


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or contains no usable text."""


class GeminiError(Exception):
    """Raised with a friendly message when the Gemini API call fails."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    """Read the Gemini API key from the environment (.env file)."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key.strip() and "your_" not in key.lower():
        return key.strip()
    return None


def get_model_name() -> str:
    """Model name can be changed in .env without touching the code."""
    return (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()


def _clean_text(text: str) -> str:
    """Tidy raw PDF text: join hyphen-broken words and squash extra spaces."""
    text = text.replace("­", "")                 # invisible soft hyphens
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # "photo-\nsynthesis"
    text = re.sub(r"\s+", " ", text)                    # newlines/tabs -> space
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Very simple sentence splitter (good enough for textbook prose)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text)
    return [p.strip() for p in parts if p.strip()]


def _normalize(text: str) -> str:
    """Lower-case and keep only letters/digits so tiny formatting
    differences do not stop a quote from matching the textbook."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 1. Extract text from the PDF (one entry per page, so we keep page numbers)
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file) -> list[dict]:
    """
    Read a PDF and return a list like:
        [{"page": 1, "page_label": "1", "text": "..."}, ...]

    `pdf_file` can be a file path or a file-like object (Streamlit upload).
    Pages without text are skipped. Raises PDFProcessingError for empty,
    encrypted, broken or scanned (image-only) PDFs.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(pdf_file)
    except Exception as exc:  # broken file, not really a PDF, etc.
        raise PDFProcessingError(
            "This file could not be opened as a PDF. It may be damaged or not a real PDF."
        ) from exc

    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise PDFProcessingError
        except Exception as exc:
            raise PDFProcessingError(
                "This PDF is password-protected. Please upload an unlocked copy."
            ) from exc

    if len(reader.pages) == 0:
        raise PDFProcessingError("This PDF has no pages.")

    # Printed page labels (e.g. "45") if the PDF defines them; else "1", "2"...
    try:
        labels = list(reader.page_labels)
    except Exception:
        labels = []

    pages = []
    for index, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""  # one bad page should not stop the whole book
        text = _clean_text(raw)
        if text:
            page_no = index + 1
            label = labels[index] if index < len(labels) else str(page_no)
            pages.append({"page": page_no, "page_label": str(label), "text": text})

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars < MIN_TEXT_CHARS:
        raise PDFProcessingError(
            "No readable text was found in this PDF. It is probably a scanned "
            "document (pictures of pages). Please upload a text-based PDF, or run "
            "OCR on it first (for example with Google Drive or Adobe 'Recognize Text')."
        )
    return pages


# ---------------------------------------------------------------------------
# 2. Split the page text into smaller overlapping chunks
# ---------------------------------------------------------------------------
def split_text_into_chunks(
    pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Split every page into chunks of about `chunk_size` characters.

    * Chunks never cross a page boundary, so every chunk has ONE exact page.
    * We cut at sentence ends, so sentences are not broken in the middle.
    * The last few sentences of a chunk are repeated at the start of the next
      one (overlap) so that an idea split across two chunks is not lost.
    """
    chunks: list[dict] = []

    for page in pages:
        sentences = []
        for sentence in _split_sentences(page["text"]):
            # A single huge "sentence" (e.g. a table) is cut into pieces.
            while len(sentence) > chunk_size:
                sentences.append(sentence[:chunk_size])
                sentence = sentence[chunk_size - overlap:]
            sentences.append(sentence)

        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if current and current_len + len(sentence) + 1 > chunk_size:
                chunks.append(_make_chunk(len(chunks), page, current))
                # keep the last sentences (up to `overlap` chars) for context
                kept: list[str] = []
                kept_len = 0
                for prev in reversed(current):
                    if kept_len + len(prev) > overlap:
                        break
                    kept.insert(0, prev)
                    kept_len += len(prev) + 1
                current, current_len = kept, kept_len
            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            chunks.append(_make_chunk(len(chunks), page, current))

    return chunks


def _make_chunk(chunk_id: int, page: dict, sentences: list[str]) -> dict:
    return {
        "chunk_id": chunk_id,
        "page": page["page"],
        "page_label": page.get("page_label", str(page["page"])),
        "text": " ".join(sentences),
    }


# ---------------------------------------------------------------------------
# 3. Turn text into embeddings (lists of numbers that capture meaning)
# ---------------------------------------------------------------------------
def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Download (first time only, ~90 MB) and load the Sentence Transformer."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def create_embeddings(texts: list[str], model) -> np.ndarray:
    """
    Convert a list of texts into a 2-D NumPy array of embeddings.
    Vectors are normalised to length 1, so the dot product between two
    vectors equals their cosine similarity.
    """
    if not texts:
        raise ValueError("There is no text to embed.")
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype="float32")


# ---------------------------------------------------------------------------
# 4. Store embeddings in a FAISS index
# ---------------------------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray):
    """
    Build an exact FAISS index. IndexFlatIP = "inner product", which is
    cosine similarity because our vectors are normalised.
    """
    import faiss

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2-D array.")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# 5. Find the chunks most similar to the student's question
# ---------------------------------------------------------------------------
def retrieve_relevant_chunks(
    question: str, index, chunks: list[dict], model, top_k: int = TOP_K
) -> list[dict]:
    """Return the `top_k` most relevant chunks, each with a similarity score."""
    if not question.strip():
        return []
    query_vector = create_embeddings([question], model)
    k = min(top_k, len(chunks))
    scores, ids = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:      # FAISS uses -1 for "no result"
            continue
        chunk = dict(chunks[int(idx)])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


# ---------------------------------------------------------------------------
# Gemini helper: one place that calls the API and handles every error
# ---------------------------------------------------------------------------
def _call_gemini(
    prompt: str,
    system_instruction: str,
    api_key: str | None,
    model_name: str | None = None,
    json_output: bool = False,
    temperature: float = 0.2,
) -> str:
    if not api_key:
        raise GeminiError(
            "Gemini API key not found. Add GEMINI_API_KEY to your .env file "
            "(see README) or paste it in the sidebar."
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise GeminiError(
            "The google-genai package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type="application/json" if json_output else "text/plain",
    )

    # Try the chosen model first; if Google has retired it, try the aliases.
    first = model_name or get_model_name()
    models_to_try = [first] + [m for m in FALLBACK_GEMINI_MODELS if m != first]

    last_error: Exception | None = None
    for name in models_to_try:
        try:
            response = client.models.generate_content(
                model=name, contents=prompt, config=config
            )
            text = (response.text or "").strip()
            if not text:
                raise GeminiError(
                    "Gemini returned an empty reply (it may have been blocked by a "
                    "safety filter). Please rephrase your question."
                )
            return text
        except GeminiError:
            raise
        except Exception as exc:  # map raw API errors to friendly messages
            message = str(exc)
            code = getattr(exc, "code", None)
            lower = message.lower()
            if "api_key_invalid" in lower or "api key not valid" in lower or code in (401, 403):
                raise GeminiError(
                    "Your Gemini API key was rejected. Check that it is copied correctly "
                    "in the .env file and that it is still active in Google AI Studio."
                ) from exc
            if code == 429 or "resource_exhausted" in lower or "quota" in lower:
                raise GeminiError(
                    "The free Gemini limit has been reached for now. Wait a minute and "
                    "try again (free keys allow a limited number of requests per minute/day)."
                ) from exc
            if code == 404 or "not found" in lower or "not supported" in lower:
                last_error = exc
                continue  # try the next model name
            if code is not None and int(code) >= 500:
                raise GeminiError(
                    "Gemini's servers are busy right now. Please try again in a moment."
                ) from exc
            if any(w in lower for w in ("connection", "timeout", "timed out", "network", "resolve")):
                raise GeminiError(
                    "Could not reach Gemini. Please check your internet connection."
                ) from exc
            raise GeminiError(f"Gemini API error: {message[:300]}") from exc

    raise GeminiError(
        f"The Gemini model '{first}' is not available for your key. Open .env and set "
        "GEMINI_MODEL to a model listed at https://ai.google.dev/gemini-api/docs/models"
    ) from last_error


def _parse_json_reply(text: str) -> dict:
    """Gemini usually returns clean JSON; this also copes with ```json fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    # Could not parse: treat the whole reply as the answer, with no quote.
    return {"found": True, "answer": text.strip(), "quote": "", "page": None}


def _verify_quote(quote: str, chunks: list[dict]) -> dict | None:
    """Return the chunk that really contains `quote`, or None if it is invented."""
    norm_quote = _normalize(quote.strip().strip(".…").strip())
    if len(norm_quote) < 15:
        return None
    for chunk in chunks:
        if norm_quote in _normalize(chunk["text"]):
            return chunk
    return None


def _closest_sentence(answer: str, question: str, chunks: list[dict]) -> tuple[str, dict] | None:
    """If Gemini's quote can't be verified, pick the real textbook sentence
    that shares the most words with the question + answer."""
    target = set(w for w in _normalize(answer + " " + question).split() if len(w) > 3)
    best, best_score = None, 0
    for chunk in chunks:
        for sentence in _split_sentences(chunk["text"]):
            if len(sentence) < 25:
                continue
            words = set(w for w in _normalize(sentence).split() if len(w) > 3)
            score = len(words & target)
            if score > best_score:
                best, best_score = (sentence, chunk), score
    return best


# ---------------------------------------------------------------------------
# 6. Generate an answer from the retrieved chunks
# ---------------------------------------------------------------------------
def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    api_key: str | None = None,
    model_name: str | None = None,
) -> dict:
    """
    Ask Gemini to answer using only the retrieved chunks.

    Returns a dict:
        answer         - Markdown answer text
        found          - False when the textbook does not contain the answer
        quote          - a sentence that REALLY appears in the textbook (or "")
        page           - PDF page number of that sentence (from our metadata)
        page_label     - printed page label of that page
        quote_verified - True if Gemini's own quote was found word-for-word;
                         False if we had to auto-select the closest real line
    """
    if not retrieved_chunks:
        return _not_found_result()

    context = "\n\n".join(
        f"[Source {i} | Page {c['page']}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, start=1)
    )
    prompt = ANSWER_PROMPT_TEMPLATE.format(
        context=context, question=question.strip(), not_found=NOT_FOUND_MESSAGE
    )
    reply = _call_gemini(
        prompt, SYSTEM_PROMPT, api_key or get_api_key(), model_name, json_output=True
    )
    data = _parse_json_reply(reply)

    answer = str(data.get("answer") or "").strip()
    found = bool(data.get("found", True))
    if not answer or NOT_FOUND_MESSAGE.lower().rstrip(".") in answer.lower():
        found = False
    if not found:
        return _not_found_result()

    # Never trust the model's quote or page blindly: check them against the PDF.
    quote = str(data.get("quote") or "").strip().strip('"“”')
    source_chunk = _verify_quote(quote, retrieved_chunks) if quote else None
    verified = source_chunk is not None

    if not verified:
        closest = _closest_sentence(answer, question, retrieved_chunks)
        if closest:
            quote, source_chunk = closest
        else:
            quote, source_chunk = "", retrieved_chunks[0]

    return {
        "answer": answer,
        "found": True,
        "quote": quote,
        "page": source_chunk["page"],
        "page_label": source_chunk.get("page_label", str(source_chunk["page"])),
        "quote_verified": verified,
    }


def _not_found_result() -> dict:
    return {
        "answer": NOT_FOUND_MESSAGE,
        "found": False,
        "quote": "",
        "page": None,
        "page_label": None,
        "quote_verified": False,
    }


# ---------------------------------------------------------------------------
# 7. Explain the previous answer in simpler words
# ---------------------------------------------------------------------------
def explain_simpler(
    previous_answer: str, api_key: str | None = None, model_name: str | None = None
) -> str:
    """Rewrite an answer in easier language without adding new information."""
    if not previous_answer or not previous_answer.strip():
        raise GeminiError("There is no answer to simplify yet. Ask a question first.")
    prompt = SIMPLIFY_PROMPT_TEMPLATE.format(answer=previous_answer.strip())
    return _call_gemini(
        prompt, SIMPLIFY_SYSTEM_PROMPT, api_key or get_api_key(), model_name,
        json_output=False, temperature=0.3,
    )


# ---------------------------------------------------------------------------
# Convenience: run steps 1-4 in one call (used by app.py)
# ---------------------------------------------------------------------------
def process_pdf(pdf_file, embedding_model) -> dict:
    """Extract -> chunk -> embed -> index. Returns everything the app needs."""
    pages = extract_text_from_pdf(pdf_file)
    chunks = split_text_into_chunks(pages)
    embeddings = create_embeddings([c["text"] for c in chunks], embedding_model)
    index = build_faiss_index(embeddings)
    return {"pages": pages, "chunks": chunks, "index": index}

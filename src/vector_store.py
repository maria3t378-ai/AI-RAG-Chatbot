
import json
import re
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_SAVE_DIR = "vectorstore"
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.30


# ==========================================
# 2. STOP WORDS
# ==========================================

STOP_WORDS = {
    "what", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "on",
    "to", "for", "and", "or", "how",
    "does", "do", "did", "can", "could",
    "would", "should", "with", "by",
    "explain", "describe", "tell", "me",
    "about", "please", "used", "use",
    "difference", "between", "applications",
    "application", "work", "works", "help",
    "helps", "step", "steps", "it",
    "define", "definition", "give", "example",
    "examples", "briefly"
}


# ==========================================
# 3. TOPIC DEFINITIONS
# ==========================================

TOPIC_PATTERNS = {
    "EMBEDDINGS": [
        r"\bembeddings?\b",
        r"\btext vectors?\b",
        r"\bvector representations?\b"
    ],

    "FAISS": [
        r"\bfaiss\b",
        r"\bsimilarity search\b"
    ],

    "RAG": [
        r"\brag\b",
        r"\bretrieval[- ]augmented generation\b"
    ],

    "DEEP_LEARNING": [
        r"\bdeep learning\b",
        r"\bdl\b"
    ],

    "MACHINE_LEARNING": [
        r"\bmachine learning\b",
        r"\bml\b"
    ],

    "ARTIFICIAL_INTELLIGENCE": [
        r"\bartificial intelligence\b",
        r"\bai\b"
    ]
}


# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================

@lru_cache(maxsize=1)
def load_embedding_model():

    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding model loaded!")

    return model


# ==========================================
# 5. METADATA HELPERS
# ==========================================

def get_chunk_value(chunk, key, default=None):

    if not isinstance(chunk, dict):
        return default

    value = chunk.get(key)

    if value is None:

        metadata = chunk.get("metadata", {})

        if isinstance(metadata, dict):
            value = metadata.get(key)

    return default if value is None else value


def normalize_chunk_metadata(chunk):

    normalized = dict(chunk)

    metadata = normalized.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    metadata = dict(metadata)

    source = get_chunk_value(
        chunk,
        "source",
        "Uploaded PDF"
    )

    page = get_chunk_value(
        chunk,
        "page",
        get_chunk_value(chunk, "page_number", "Unknown")
    )

    chunk_number = get_chunk_value(
        chunk,
        "chunk",
        get_chunk_value(chunk, "chunk_number", None)
    )

    normalized["source"] = str(source)
    normalized["page"] = page

    metadata["source"] = str(source)
    metadata["page"] = page

    if chunk_number is not None:
        normalized["chunk"] = chunk_number
        metadata["chunk"] = chunk_number

    normalized["metadata"] = metadata

    return normalized


# ==========================================
# 6. TEXT NORMALIZATION
# ==========================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9\s-]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 7. KEYWORD EXTRACTION
# ==========================================

def extract_keywords(text):

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        str(text).lower()
    )

    return {
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


# ==========================================
# 8. TOPIC DETECTION
# ==========================================

def detect_topics(text):

    text = normalize_text(text)

    detected_topics = set()

    for topic, patterns in TOPIC_PATTERNS.items():

        for pattern in patterns:

            if re.search(pattern, text):
                detected_topics.add(topic)
                break

    return detected_topics


# ==========================================
# 9. TOPIC MATCHING
# ==========================================

def calculate_topic_score(question_topics, chunk_topics):

    if not question_topics:
        return 0.0

    matched_topics = question_topics.intersection(
        chunk_topics
    )

    return len(matched_topics) / len(question_topics)


def has_exact_topic_match(question, chunk_text):

    question_topics = detect_topics(question)
    chunk_topics = detect_topics(chunk_text)

    return bool(question_topics.intersection(chunk_topics))


# ==========================================
# 10. KEYWORD MATCHING
# ==========================================

def calculate_keyword_score(question, chunk_text):

    question_keywords = extract_keywords(question)
    chunk_keywords = extract_keywords(chunk_text)

    if not question_keywords:
        return 0.0

    matched_keywords = question_keywords.intersection(
        chunk_keywords
    )

    return len(matched_keywords) / len(question_keywords)


# ==========================================
# 11. CREATE VECTOR DATABASE
# ==========================================

def create_vector_store(
    chunks,
    save_dir=DEFAULT_SAVE_DIR
):

    if not chunks:
        raise ValueError("No text chunks provided.")

    valid_chunks = []
    seen_chunks = set()

    for chunk in chunks:

        if not isinstance(chunk, dict):
            continue

        text = str(
            get_chunk_value(chunk, "text", "")
        ).strip()

        if not text:
            continue

        normalized_text = normalize_text(text)

        if normalized_text in seen_chunks:
            continue

        seen_chunks.add(normalized_text)

        valid_chunk = normalize_chunk_metadata(chunk)

        valid_chunk["text"] = text

        valid_chunks.append(valid_chunk)

    if not valid_chunks:
        raise ValueError("No valid text found in chunks.")

    save_path = Path(save_dir)

    save_path.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in valid_chunks
    ]

    print(f"Creating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    if embeddings.ndim != 2:
        raise ValueError("Invalid embedding dimensions.")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings contain invalid values.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    index_file = save_path / "faiss_index.bin"

    faiss.write_index(
        index,
        str(index_file)
    )

    metadata = {
        "chunks": valid_chunks,
        "model": EMBEDDING_MODEL,
        "dimension": int(dimension),
        "total_chunks": len(valid_chunks)
    }

    metadata_file = save_path / "metadata.json"

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("Vector database created successfully.")

    return index, valid_chunks


# ==========================================
# 12. LOAD VECTOR DATABASE
# ==========================================

def load_vector_store(save_dir=DEFAULT_SAVE_DIR):

    save_path = Path(save_dir)

    index_file = save_path / "faiss_index.bin"
    metadata_file = save_path / "metadata.json"

    if not index_file.exists() or not metadata_file.exists():

        raise FileNotFoundError(
            "Vector database not found. Process PDFs first."
        )

    index = faiss.read_index(str(index_file))

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Invalid vector database metadata.")

    if metadata.get("model") != EMBEDDING_MODEL:

        raise ValueError(
            "Embedding model does not match saved database."
        )

    chunks = metadata.get("chunks", [])

    if not isinstance(chunks, list):
        raise ValueError("Invalid chunk metadata.")

    if index.ntotal != len(chunks):

        raise ValueError(
            "FAISS index and metadata do not match. "
            "Please process PDFs again."
        )

    if index.ntotal == 0:
        raise ValueError("Vector database is empty.")

    saved_dimension = metadata.get("dimension")

    if saved_dimension is not None and index.d != saved_dimension:
        raise ValueError("Embedding dimensions do not match.")

    restored_chunks = []

    for chunk in chunks:

        if not isinstance(chunk, dict):
            raise ValueError("Invalid chunk found.")

        restored_chunks.append(
            normalize_chunk_metadata(chunk)
        )

    return index, restored_chunks


# ==========================================
# 13. HYBRID SEARCH
# ==========================================

def search_similar_chunks(
    question,
    index,
    chunks,
    top_k=DEFAULT_TOP_K,
    min_score=DEFAULT_MIN_SCORE
):

    if not question or not str(question).strip():
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1.")

    if index is None or index.ntotal == 0:
        return []

    if not chunks:
        return []

    if index.ntotal != len(chunks):

        raise ValueError(
            "Index and chunks do not match. "
            "Please process PDFs again."
        )

    model = load_embedding_model()

    # Question embedding
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    if (
        query_embedding.ndim != 2
        or query_embedding.shape[1] != index.d
    ):
        raise ValueError("Question embedding dimension mismatch.")

    # Search all indexed chunks
    semantic_scores, indices = index.search(
        query_embedding,
        index.ntotal
    )

    question_keywords = extract_keywords(question)
    question_topics = detect_topics(question)

    results = []

    # Evaluate every candidate
    for semantic_score, idx in zip(
        semantic_scores[0],
        indices[0]
    ):

        if idx < 0 or idx >= len(chunks):
            continue

        chunk = normalize_chunk_metadata(chunks[int(idx)])

        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        semantic_score = float(
            np.clip(semantic_score, 0.0, 1.0)
        )

        keyword_score = calculate_keyword_score(
            question,
            text
        )

        chunk_topics = detect_topics(text)

        topic_score = calculate_topic_score(
            question_topics,
            chunk_topics
        )

        exact_topic_match = bool(
            question_topics.intersection(chunk_topics)
        )

        chunk_keywords = extract_keywords(text)

        has_keyword_match = bool(
            question_keywords.intersection(chunk_keywords)
        )

        # Hybrid relevance score
        combined_score = (
            0.55 * semantic_score
            + 0.30 * keyword_score
            + 0.15 * topic_score
        )

        # Small bonus for matching both topic and keyword
        if exact_topic_match and has_keyword_match:
            combined_score += 0.03

        combined_score = min(combined_score, 1.0)

        # Strict topic filter:
        # If a question has a recognized topic,
        # a chunk must match that topic.
        if question_topics and not exact_topic_match:
            continue

        # General relevance filter
        is_relevant = (
            combined_score >= min_score
            and (
                has_keyword_match
                or semantic_score >= 0.45
            )
        )

        if not is_relevant:
            continue

        result = dict(chunk)

        result.update({
            "score": float(combined_score),
            "semantic_score": float(semantic_score),
            "keyword_score": float(keyword_score),
            "topic_score": float(topic_score),
            "topics": sorted(chunk_topics),
            "faiss_position": int(idx),
            "exact_topic_match": exact_topic_match
        })

        results.append(result)

    # Highest relevance first
    results.sort(
        key=lambda item: (
            item["score"],
            item["semantic_score"],
            item["keyword_score"]
        ),
        reverse=True
    )

    # Remove duplicate chunks
    final_results = []
    seen = set()

    for result in results:

        source = str(
            result.get("source", "")
        ).casefold()

        page = str(
            result.get("page", "")
        ).casefold()

        text_key = normalize_text(
            result.get("text", "")
        )

        unique_key = (
            source,
            page,
            text_key
        )

        if unique_key in seen:
            continue

        seen.add(unique_key)

        final_results.append(result)

        if len(final_results) >= top_k:
            break

    return final_results

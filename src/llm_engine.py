
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 180

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "and", "or",
    "of", "in", "on", "to", "for", "with", "how", "does",
    "do", "can", "could", "would", "should", "explain",
    "tell", "me", "about", "define", "difference",
    "between", "which", "why", "when", "where", "it"
}


# =========================================================
# 2. TEXT CLEANING HELPERS
# =========================================================

def clean_text(text):
    """Clean PDF text while preserving meaningful formatting."""

    if not text:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Join words split by a hyphen at a PDF line break.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Normalize spaces but preserve paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)

    return text.strip()


def normalize(text):
    """Normalize text for matching."""

    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_keywords(text):
    """Extract useful keywords."""

    words = re.findall(r"\b[a-zA-Z0-9]+\b", normalize(text))

    return {
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


# =========================================================
# 3. CHUNK HELPERS
# =========================================================

def get_chunk_text(chunk):
    """Read text from a chunk dictionary or object."""

    if isinstance(chunk, str):
        return clean_text(chunk)

    if isinstance(chunk, dict):
        return clean_text(
            chunk.get("text", chunk.get("page_content", ""))
        )

    if hasattr(chunk, "page_content"):
        return clean_text(chunk.page_content)

    return ""


def get_chunk_metadata(chunk):
    """Read metadata from a chunk."""

    if isinstance(chunk, dict):
        return chunk.get("metadata", chunk)

    if hasattr(chunk, "metadata"):
        return chunk.metadata or {}

    return {}


def get_chunk_source_page(chunk):
    """Get source and page information for overlap checking."""

    metadata = get_chunk_metadata(chunk)

    source = (
        metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("filename")
    )

    page = metadata.get("page", metadata.get("page_number"))

    return source, page


def merge_overlapping_text(previous_text, next_text, max_overlap=500):
    """
    Merge two consecutive chunks by removing their repeated overlap.

    Only exact matching overlap is removed. If no reliable overlap
    is found, the two chunks are joined normally.
    """

    previous_text = clean_text(previous_text)
    next_text = clean_text(next_text)

    if not previous_text:
        return next_text

    if not next_text:
        return previous_text

    # Avoid adding an identical chunk again.
    if next_text in previous_text:
        return previous_text

    if previous_text in next_text:
        return next_text

    maximum = min(len(previous_text), len(next_text), max_overlap)

    # Find the longest exact suffix/prefix match.
    for overlap_size in range(maximum, 29, -1):
        if previous_text[-overlap_size:] == next_text[:overlap_size]:
            remaining_text = next_text[overlap_size:].lstrip()

            if remaining_text:
                return previous_text.rstrip() + " " + remaining_text

            return previous_text

    return previous_text.rstrip() + "\n\n" + next_text.lstrip()


def collect_document_text(chunks):
    """
    Combine document chunks and remove repeated overlap.

    Chunks are merged only when they belong to the same source
    and page. This prevents unrelated pages from being merged.
    """

    if not chunks:
        return ""

    if isinstance(chunks, (str, dict)):
        chunks = [chunks]

    merged_parts = []
    previous_text = None
    previous_source_page = None

    for chunk in chunks:
        current_text = get_chunk_text(chunk)

        if not current_text:
            continue

        current_source_page = get_chunk_source_page(chunk)

        if previous_text is None:
            previous_text = current_text
            previous_source_page = current_source_page
            continue

        same_source = (
            previous_source_page[0] == current_source_page[0]
        )

        same_page = (
            previous_source_page[1] == current_source_page[1]
        )

        # If metadata is missing, do not assume chunks overlap.
        metadata_available = (
            previous_source_page[0] is not None
            and current_source_page[0] is not None
            and previous_source_page[1] is not None
            and current_source_page[1] is not None
        )

        if metadata_available and same_source and same_page:
            previous_text = merge_overlapping_text(
                previous_text,
                current_text
            )
        else:
            merged_parts.append(previous_text)
            previous_text = current_text

        previous_source_page = current_source_page

    if previous_text:
        merged_parts.append(previous_text)

    return clean_text("\n\n".join(merged_parts))


def select_relevant_chunks(question, chunks, max_chunks=3):
    """Select chunks with the strongest keyword overlap."""

    if not chunks:
        return []

    if isinstance(chunks, (str, dict)):
        chunks = [chunks]

    question_keywords = get_keywords(question)

    if not question_keywords:
        return list(chunks)[:max_chunks]

    scored_chunks = []

    for index, chunk in enumerate(chunks):
        chunk_keywords = get_keywords(get_chunk_text(chunk))
        overlap = question_keywords.intersection(chunk_keywords)

        score = len(overlap) / max(len(question_keywords), 1)

        scored_chunks.append((score, index, chunk))

    scored_chunks.sort(
        key=lambda item: (item[0], -item[1]),
        reverse=True
    )

    selected = [
        item[2]
        for item in scored_chunks[:max_chunks]
        if item[0] > 0
    ]

    if not selected:
        return list(chunks)[:max_chunks]

    return selected


# =========================================================
# 4. DOCUMENT SECTION EXTRACTION
# =========================================================

def extract_section(text, start_pattern, end_pattern=None):
    """Extract content between two section headings."""

    text = clean_text(text)

    if not text:
        return None

    flags = re.IGNORECASE | re.MULTILINE

    start_match = re.search(start_pattern, text, flags)

    if not start_match:
        return None

    section_start = start_match.end()

    if end_pattern:
        end_match = re.search(
            end_pattern,
            text[section_start:],
            flags
        )

        if end_match:
            section_end = section_start + end_match.start()
            section = text[section_start:section_end]
        else:
            section = text[section_start:]
    else:
        section = text[section_start:]

    section = clean_text(section)

    return section if section else None


def format_section(title, content):
    """Format an extracted section."""

    content = clean_text(content)

    if not content:
        return None

    return f"**{title}**\n\n{content}"


# =========================================================
# 5. ARTIFICIAL INTELLIGENCE EXTRACTORS
# =========================================================

def extract_ai_definition(text):
    section = extract_section(
        text,
        r"\b1\s*\.\s*Artificial\s+Intelligence\s*\(\s*AI\s*\)",
        r"Applications\s+of\s+AI\s*:"
    )

    return format_section("Artificial Intelligence", section)


def extract_ai_applications(text):
    section = extract_section(
        text,
        r"Applications\s+of\s+AI\s*:",
        r"\b2\s*\.\s*Machine\s+Learning"
    )

    return format_section("Applications of AI", section)


# =========================================================
# 6. MACHINE LEARNING EXTRACTORS
# =========================================================

def extract_ml_definition(text):
    section = extract_section(
        text,
        r"\b2\s*\.\s*Machine\s+Learning\s*\(\s*ML\s*\)",
        r"Types\s+of\s+Machine\s+Learning\s*:"
    )

    return format_section("Machine Learning", section)


def extract_ml_types(text):
    section = extract_section(
        text,
        r"Types\s+of\s+Machine\s+Learning\s*:",
        r"Applications\s+of\s+ML\s*:"
    )

    return format_section("Types of Machine Learning", section)


# =========================================================
# 7. DEEP LEARNING EXTRACTOR
# =========================================================

def extract_deep_learning(text):
    section = extract_section(
        text,
        r"\b3\s*\.\s*Deep\s+Learning\s*\(\s*DL\s*\)",
        r"\b4\s*\.\s*Retrieval[-\s]+Augmented\s+Generation"
    )

    return format_section("Deep Learning", section)


# =========================================================
# 8. RAG EXTRACTORS
# =========================================================

def extract_rag_definition(text):
    section = extract_section(
        text,
        r"\b4\s*\.\s*Retrieval[-\s]+Augmented\s+Generation\s*\(\s*RAG\s*\)",
        r"How\s+RAG\s+Works\s*:"
    )

    return format_section(
        "Retrieval-Augmented Generation (RAG)",
        section
    )


def extract_rag_steps(text):
    section = extract_section(
        text,
        r"How\s+RAG\s+Works\s*:",
        r"\b5\s*\.\s*FAISS"
    )

    if not section:
        return None

    # Remove a repeated heading.
    section = re.sub(
        r"^\s*How\s+RAG\s+Works\s*:?\s*",
        "",
        section,
        flags=re.IGNORECASE
    )

    # Stop before Applications of RAG if it appears in the section.
    section = re.split(
        r"Applications\s+of\s+RAG\s*:",
        section,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    # Remove a stray next-section number at the end.
    section = re.sub(r"\s*\b5\s*\.\s*$", "", section)

    return format_section("How RAG Works", section)


# =========================================================
# 9. FAISS EXTRACTOR
# =========================================================

def extract_faiss(text):
    section = extract_section(
        text,
        r"\b5\s*\.\s*FAISS\b",
        r"\b6\s*\.\s*Embeddings\b"
    )

    if not section:
        return None

    # Remove repeated FAISS heading inside the extracted section.
    section = re.sub(
        r"^\s*FAISS\s*\(\s*Facebook\s+AI\s+Similarity\s+Search\s*\)\s*",
        "",
        section,
        flags=re.IGNORECASE
    )

    return format_section(
        "FAISS (Facebook AI Similarity Search)",
        section
    )


# =========================================================
# 10. EMBEDDINGS EXTRACTOR
# =========================================================

def extract_embeddings(text):
    """Extract only the numbered Embeddings section."""

    if not text:
        return None

    section = extract_section(
        text,
        r"\b6\s*\.\s*Embeddings\b",
        r"\b7\s*\.\s*Difference\s+Between\s+AI"
    )

    if not section:
        return None

    # Remove a repeated heading.
    section = re.sub(
        r"^\s*Embeddings\s*:?\s*",
        "",
        section,
        flags=re.IGNORECASE
    )

    # Remove a stray next-section number.
    section = re.sub(r"\s*\b7\s*\.\s*$", "", section)

    section = clean_text(section)

    # Confirm that the actual definition is present.
    if "convert text" not in normalize(section):
        return None

    return format_section("Embeddings", section)


# =========================================================
# 11. AI, ML AND DL COMPARISON EXTRACTOR
# =========================================================

def extract_ai_ml_dl_comparison(text):
    section = extract_section(
        text,
        r"\b7\s*\.\s*Difference\s+Between\s+AI\s*,?\s*ML\s*,?\s*and\s*DL",
        r"\b8\s*\.\s*Conclusion"
    )

    return format_section(
        "Difference Between AI, ML and DL",
        section
    )


# =========================================================
# 12. INTENT DETECTION
# =========================================================

def detect_intent(question):
    """Identify what the user is asking."""

    q = normalize(question)

    # Comparison
    if (
        ("difference" in q or "compare" in q or "comparison" in q)
        and "ai" in q
        and ("ml" in q or "machine learning" in q)
        and ("dl" in q or "deep learning" in q)
    ):
        return "comparison"

    # AI applications
    if (
        "application" in q
        or "applications" in q
        or "uses" in q
        or "use cases" in q
    ) and ("ai" in q or "artificial intelligence" in q):
        return "ai_applications"

    # AI definition
    if "artificial intelligence" in q or re.search(r"\bai\b", q):
        return "ai_definition"

    # Machine Learning types
    if (
        ("types" in q or "type" in q)
        and ("machine learning" in q or re.search(r"\bml\b", q))
    ):
        return "ml_types"

    # Machine Learning definition
    if "machine learning" in q or re.search(r"\bml\b", q):
        return "ml_definition"

    # Deep Learning
    if "deep learning" in q or re.search(r"\bdl\b", q):
        return "deep_learning"

    # RAG steps
    if (
        ("how" in q or "steps" in q or "workflow" in q)
        and ("rag" in q or "retrieval" in q)
    ):
        return "rag_steps"

    # RAG definition
    if "rag" in q or "retrieval augmented generation" in q:
        return "rag_definition"

    # FAISS
    if "faiss" in q or "similarity search" in q:
        return "faiss"

    # Embeddings
    if "embedding" in q or "vector" in q:
        return "embeddings"

    return "general"


# =========================================================
# 13. STRUCTURED ANSWER ROUTER
# =========================================================

def extract_document_answer(question, chunks, all_chunks=None):
    """Extract a structured answer from the uploaded document."""

    source_chunks = all_chunks if all_chunks else chunks
    document_text = collect_document_text(source_chunks)

    if not document_text:
        return None

    intent = detect_intent(question)

    extractors = {
        "ai_definition": extract_ai_definition,
        "ai_applications": extract_ai_applications,
        "ml_definition": extract_ml_definition,
        "ml_types": extract_ml_types,
        "deep_learning": extract_deep_learning,
        "rag_definition": extract_rag_definition,
        "rag_steps": extract_rag_steps,
        "faiss": extract_faiss,
        "embeddings": extract_embeddings,
        "comparison": extract_ai_ml_dl_comparison,
    }

    extractor = extractors.get(intent)

    if extractor:
        return extractor(document_text)

    return None


# =========================================================
# 14. ANSWER CLEANING
# =========================================================

def clean_answer(answer):
    """Clean generated output without damaging Markdown."""

    if not answer:
        return ""

    answer = str(answer).strip()

    answer = re.sub(
        r"^(Answer|Response)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )

    answer = answer.replace("\r\n", "\n")
    answer = answer.replace("\r", "\n")

    # Preserve paragraph breaks and Markdown formatting.
    answer = re.sub(r"[ \t]+", " ", answer)
    answer = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", answer)

    return answer.strip()


# =========================================================
# 15. SOURCE INFORMATION
# =========================================================

def get_source_information(chunk, index=1):
    """Get source filename and page number."""

    metadata = get_chunk_metadata(chunk)

    source = (
        metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("filename")
        or "Uploaded document"
    )

    page = metadata.get("page", metadata.get("page_number"))

    if page is not None:
        try:
            page = int(page)

            # PyPDF commonly uses zero-based page indexes.
            if page == 0:
                page = 1

        except (ValueError, TypeError):
            pass

    return {
        "index": index,
        "source": str(source),
        "page": page
    }


def build_citation_sources(chunks):
    """Build unique source references."""

    sources = []
    seen = set()

    if not chunks:
        return sources

    for chunk in chunks:
        info = get_source_information(chunk)
        key = (info["source"], info["page"])

        if key in seen:
            continue

        seen.add(key)
        info["index"] = len(sources) + 1
        sources.append(info)

    return sources


# =========================================================
# 16. RAG ANSWER GENERATOR
# =========================================================

class RAGAnswerGenerator:
    """Generate answers using structured extraction and FLAN-T5."""

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.model_error = None

    # -----------------------------------------------------
    # Load model only when needed
    # -----------------------------------------------------

    def load_model(self):
        if self.model_loaded:
            return True

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name
            )

            self.model.eval()

            self.model_loaded = True
            self.model_error = None

            return True

        except Exception as error:
            self.model_error = str(error)
            self.model_loaded = False

            return False

    # -----------------------------------------------------
    # Prepare context
    # -----------------------------------------------------

    def prepare_context(self, question, chunks, max_chunks=4):
        selected_chunks = select_relevant_chunks(
            question,
            chunks,
            max_chunks=max_chunks
        )

        context_parts = []

        for index, chunk in enumerate(selected_chunks, start=1):
            text = get_chunk_text(chunk)

            if not text:
                continue

            info = get_source_information(chunk, index)

            context_parts.append(
                f"Source {index}:\n"
                f"Document: {info['source']}\n"
                f"Page: {info['page']}\n"
                f"Content: {text}"
            )

        return "\n\n".join(context_parts), selected_chunks

    # -----------------------------------------------------
    # Build FLAN-T5 prompt
    # -----------------------------------------------------

    def build_prompt(self, question, context):
        return f"""
Answer the question using only the information in the context.

If the answer is not available in the context, say:
"The uploaded documents do not contain enough information to answer this question."

Write a clear, accurate and concise answer.
Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    # -----------------------------------------------------
    # Check answer relevance
    # -----------------------------------------------------

    def is_answer_relevant(self, question, answer):
        if not answer:
            return False

        normalized_answer = normalize(answer)

        if len(normalized_answer) < 8:
            return False

        if "does not contain enough information" in normalized_answer:
            return True

        question_keywords = get_keywords(question)
        answer_keywords = get_keywords(answer)

        if not question_keywords:
            return True

        return bool(question_keywords.intersection(answer_keywords))

    # -----------------------------------------------------
    # Generate using FLAN-T5
    # -----------------------------------------------------

    def generate_with_llm(self, question, chunks):
        if not chunks:
            return None

        context, selected_chunks = self.prepare_context(
            question,
            chunks
        )

        if not context:
            return None

        if not self.load_model():
            return None

        prompt = self.build_prompt(question, context)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_LENGTH
            )

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=4,
                    early_stopping=True
                )

            answer = self.tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )

            answer = clean_answer(answer)

            if self.is_answer_relevant(question, answer):
                return answer

            return None

        except Exception as error:
            self.model_error = str(error)
            return None

    # -----------------------------------------------------
    # Extract answer from document
    # -----------------------------------------------------

    def extract_document_answer(
        self,
        question,
        chunks,
        all_chunks=None
    ):
        return extract_document_answer(
            question,
            chunks,
            all_chunks=all_chunks
        )

    # -----------------------------------------------------
    # Main answer method - compatible with app.py
    # -----------------------------------------------------

    def generate_answer(self, question, chunks, all_chunks=None):
        """Main method called by the Streamlit app."""

        if not question or not str(question).strip():
            return "Please enter a question."

        if not chunks:
            return (
                "No relevant information was retrieved. "
                "Please upload a PDF document and try again."
            )

        # First try structured document extraction.
        structured_answer = self.extract_document_answer(
            question,
            chunks,
            all_chunks=all_chunks
        )

        if structured_answer:
            return clean_answer(structured_answer)

        # If no structured section is found, use FLAN-T5.
        llm_answer = self.generate_with_llm(
            question,
            chunks
        )

        if llm_answer:
            return clean_answer(llm_answer)

        # Safe fallback: show retrieved document text.
        context, selected_chunks = self.prepare_context(
            question,
            chunks,
            max_chunks=2
        )

        if context:
            fallback_text = "\n\n".join(
                get_chunk_text(chunk)
                for chunk in selected_chunks
                if get_chunk_text(chunk)
            )

            if fallback_text:
                return (
                    "**Relevant information from your documents:**\n\n"
                    + clean_text(fallback_text)
                )

        return (
            "I could not generate a reliable answer from the "
            "retrieved document information."
        )

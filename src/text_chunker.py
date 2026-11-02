
import re


def clean_text(text):
    """
    Normalize extracted text before sentence splitting.
    """

    if not text:
        return ""

    text = re.sub(r"\s+", " ", str(text))

    return text.strip()


def split_into_sentences(text):
    """
    Split text into sentences using punctuation.
    """

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_long_sentence(sentence, chunk_size):
    """
    Split exceptionally long sentences at word boundaries.
    """

    words = sentence.split()

    parts = []
    current_part = []
    current_length = 0

    for word in words:

        additional_length = len(word) + (
            1 if current_part else 0
        )

        if (
            current_length + additional_length > chunk_size
            and current_part
        ):

            parts.append(" ".join(current_part))

            current_part = [word]
            current_length = len(word)

        else:

            current_part.append(word)
            current_length += additional_length

    if current_part:
        parts.append(" ".join(current_part))

    return parts


def split_text_into_chunks(
    extracted_pages,
    chunk_size=800,
    chunk_overlap=150
):
    """
    Create sentence-aware chunks while preserving
    page numbers, source filenames and chunk IDs.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if (
        chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError(
            "chunk_overlap must be non-negative "
            "and smaller than chunk_size."
        )

    chunks = []

    for page_data in extracted_pages:

        # Safely extract page information
        text = clean_text(
            page_data.get("text", "")
        )

        page_number = page_data.get("page", 1)

        source = page_data.get(
            "source",
            "Uploaded PDF"
        )

        if not text:
            continue

        # Split text into sentences
        sentences = split_into_sentences(text)

        # Handle exceptionally long sentences
        processed_sentences = []

        for sentence in sentences:

            if len(sentence) > chunk_size:

                processed_sentences.extend(
                    split_long_sentence(
                        sentence,
                        chunk_size
                    )
                )

            else:

                processed_sentences.append(sentence)

        start = 0

        while start < len(processed_sentences):

            current_chunk = []
            current_length = 0
            end = start

            # Build chunk within the character limit
            while end < len(processed_sentences):

                sentence = processed_sentences[end]

                extra_length = len(sentence) + (
                    1 if current_chunk else 0
                )

                if (
                    current_length + extra_length > chunk_size
                    and current_chunk
                ):
                    break

                current_chunk.append(sentence)

                current_length += extra_length

                end += 1

            chunk_text = " ".join(
                current_chunk
            ).strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page_number,
                    "source": source,
                    "chunk": len(chunks) + 1
                })

            if end >= len(processed_sentences):
                break

            # Calculate sentence-based overlap
            next_start = end
            overlap_length = 0

            while next_start > start + 1:

                previous_sentence = (
                    processed_sentences[next_start - 1]
                )

                additional_length = (
                    len(previous_sentence) + 1
                )

                if (
                    overlap_length + additional_length
                    > chunk_overlap
                ):
                    break

                overlap_length += additional_length

                next_start -= 1

            # Prevent infinite loops
            if next_start <= start:
                next_start = end

            start = next_start

    return chunks

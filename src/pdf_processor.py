
import re
from pathlib import Path
from io import BytesIO
from typing import Union

from pypdf import PdfReader


def clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving readable spacing.
    """

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Join words split by PDF line wrapping.
    # Example: artifi-
    #          cial -> artificial
    text = re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "",
        text
    )

    # Normalize spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize line breaks
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
    ]

    # Remove empty lines
    lines = [
        line for line in lines
        if line
    ]

    # Join lines into readable text
    cleaned_text = " ".join(lines)

    # Remove unnecessary multiple spaces
    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text
    )

    return cleaned_text.strip()


def extract_text_from_pdf(
    pdf_path: Union[str, Path, BytesIO],
    source: str = None
) -> list:
    """
    Extract readable text from every PDF page.

    Supports:
    - File paths
    - Path objects
    - BytesIO PDF uploads

    Returns:
        [
            {
                "page": 1,
                "text": "...",
                "source": "document.pdf"
            }
        ]
    """

    # Support uploaded PDF streams
    if isinstance(pdf_path, BytesIO):

        pdf_path.seek(0)

        reader = PdfReader(pdf_path)

        if source is None:
            source = "Uploaded PDF"

    else:

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if not pdf_path.is_file():
            raise ValueError(
                f"Invalid PDF file path: {pdf_path}"
            )

        reader = PdfReader(str(pdf_path))

        if source is None:
            source = pdf_path.name

    # Check encrypted PDF
    if reader.is_encrypted:

        try:
            result = reader.decrypt("")

            if result == 0:
                raise ValueError(
                    "This PDF is password protected."
                )

        except Exception as error:

            raise ValueError(
                f"Unable to decrypt PDF: {error}"
            )

    extracted_pages = []

    # Extract text page by page
    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            raw_text = page.extract_text() or ""

            cleaned_text = clean_pdf_text(
                raw_text
            )

            if cleaned_text:

                extracted_pages.append({
                    "page": page_number,
                    "text": cleaned_text,
                    "source": source
                })

        except Exception as error:

            raise ValueError(
                f"Error extracting page {page_number}: {error}"
            )

    # Check extracted content
    if not extracted_pages:

        raise ValueError(
            "No readable text found. "
            "This PDF may be scanned or image-based."
        )

    return extracted_pages

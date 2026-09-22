"""
data.py
-------
Responsible for ONE job: loading policy text from files on disk and
turning it into a list of small searchable chunks.

Supported formats:
    .txt   plain text (one paragraph / sentence per line)
    .pdf   text-based PDFs   (needs: pip install pypdf)
    .docx  Word documents    (needs: pip install python-docx)

Where files are read from:
    - By default: the  app/documents/  folder (next to this file).
    - Override with the DOCUMENTS_DIR environment variable, or call
      load_documents("path/to/folder_or_file") yourself.

Section headings
    A long document (e.g. "1. Premium Daycare + Preschool ... Budget
    ■15–25L ...") gets split into several small chunks. Without help, a
    chunk deep in that section ("Budget ■15–25L Revenue ■4–7L/month...")
    has no idea which business it's talking about, so a query mentioning
    "Premium Daycare" scores it no higher than a chunk from a completely
    different section. To fix that, every chunk is prefixed with the
    heading of the section it came from: "Premium Daycare + Preschool.
    Budget ■15–25L Revenue ■4–7L/month...".

    Headings are detected by:
      .docx  a paragraph using Word's "Heading"/"Title" style
      .pdf / .txt   a line matching `heading_pattern` (default: a numbered
             heading like "6. Medical Patient Accommodation" - short,
             no sentence-ending punctuation)
    This is a heuristic, not real document structure. Documents that use a
    different heading convention (e.g. ALL-CAPS lines, markdown "#") won't
    be picked up by the default pattern; pass your own `heading_pattern` to
    load_documents() if needed.

Output shape (same as the old static list, plus three extra fields):
    [{"id": 1, "text": "Premium Daycare + Preschool. Budget ■15-25L...",
      "heading": "Premium Daycare + Preschool", "source": "plan.pdf", "page": 2}, ...]

main.py only uses doc["text"], so nothing else has to change:
    from app.data import Documents
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
DEFAULT_DOCS_DIR = Path(os.getenv("DOCUMENTS_DIR", Path(__file__).parent / "documents"))

# Chunking settings.
# Long text is split into chunks of at most MAX_CHUNK_CHARS characters, because
# one embedding per giant document blurs its meaning. Chunks shorter than
# MIN_CHUNK_CHARS (page numbers, stray headings) are dropped as noise.
MAX_CHUNK_CHARS = 500
MIN_CHUNK_CHARS = 20

# A line like "6. Medical Patient Accommodation": starts with "N. ", is short,
# and (unlike a numbered sentence such as "3. Submit expenses within 30 days.")
# contains no sentence-ending punctuation of its own.
DEFAULT_HEADING_PATTERN = re.compile(r"^\d{1,2}\.\s+[^.?!]{2,78}$")

# Internal marker _read_docx uses to flag a paragraph as a heading (from its
# Word style) so the shared section-tracking logic in _sectionize can treat
# .docx the same way as .pdf/.txt without needing to know about docx styles.
_HEADING_MARK = "\x01HEADING\x01"
_HEADING_STYLE_PREFIXES = ("heading", "title")


# ---------------------------------------------------------------------------
# File readers: each returns a list of (page_number_or_None, raw_text) tuples
# ---------------------------------------------------------------------------

def _read_txt(path: Path) -> list[tuple[Optional[int], str]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return [(None, text)]


def _fix_undecoded_currency_symbol(text: str) -> str:
    """
    Some PDFs embed a currency symbol (commonly ₹) in a custom font that
    pypdf can't map to a real character, so it falls back to U+25A0 BLACK
    SQUARE ('■') wherever that symbol appeared. A stray '■' means nothing to
    an embedding model, which quietly weakens matches for money-related
    queries ("budget", "investment", "cost") on every figure in the document.

    This is a narrow, safe fix: '■' is only ever used this way immediately
    before a digit (e.g. "■15–25L"), never as a bullet or standalone symbol,
    so only that exact pattern is rewritten. If a future document legitimately
    uses '■' as a bullet before a number, this would mis-rewrite it - check
    a document's `/health` or a sample chunk if numbers look off after a
    reindex.
    """
    return re.sub(r"■(?=\d)", "₹", text)


def _read_pdf(path: Path) -> list[tuple[Optional[int], str]]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("Reading PDFs needs pypdf: pip install pypdf") from e

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty password; try that.
        if reader.decrypt("") == 0:
            raise ValueError("PDF is password-protected")

    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_no, _fix_undecoded_currency_symbol(text)))
    return pages


def _read_docx(path: Path) -> list[tuple[Optional[int], str]]:
    try:
        from docx import Document
        from docx.table import Table
    except ImportError as e:
        raise ImportError("Reading Word files needs python-docx: pip install python-docx") from e

    doc = Document(str(path))
    lines = []
    # iter_inner_content() yields paragraphs AND tables in document order.
    for block in doc.iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
        else:
            text = block.text.strip()
            if not text:
                continue
            style_name = (block.style.name or "").lower() if block.style else ""
            if style_name.startswith(_HEADING_STYLE_PREFIXES):
                lines.append(_HEADING_MARK + text)  # tag so _sectionize can find it
            else:
                lines.append(text)
    return [(None, "\n".join(lines))]


_READERS = {
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


# ---------------------------------------------------------------------------
# Sectionizing: split raw text into (heading, unit_text) pairs, tracking the
# most recent heading seen so it can be prefixed onto every chunk below it.
# ---------------------------------------------------------------------------

def _sectionize(
    text: str, ext: str, heading_pattern: re.Pattern = DEFAULT_HEADING_PATTERN
) -> list[tuple[Optional[str], str]]:
    """
    Walk raw text top-to-bottom and return (heading, unit_text) pairs, where
    `heading` is whatever section heading was most recently seen (or None,
    before the first one / if the document has none).

    - .pdf: text is hard-wrapped mid-sentence, so body lines are grouped into
      paragraphs at blank-line boundaries; a line matching `heading_pattern`
      updates the running heading instead of being merged into a paragraph.
    - .txt / .docx: every non-empty line is its own unit. For .txt, a line
      matching `heading_pattern` updates the heading. For .docx, a line
      tagged by _read_docx with _HEADING_MARK (a Word heading/title style)
      updates the heading; `heading_pattern` is also checked as a fallback
      for documents that use numbered headings without a heading style.
    """
    heading: Optional[str] = None
    result: list[tuple[Optional[str], str]] = []

    def as_heading_text(line: str) -> str:
        # "6. Medical Patient Accommodation" -> "Medical Patient Accommodation"
        return line.split(".", 1)[1].strip() if "." in line else line

    if ext == ".pdf":
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                para = re.sub(r"\s+", " ", " ".join(buffer)).strip()
                if para:
                    result.append((heading, para))
                buffer.clear()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if heading_pattern.match(line):
                flush()
                heading = as_heading_text(line)
                continue
            buffer.append(line)
        flush()

    else:  # .txt and .docx: one line = one unit
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if ext == ".docx" and line.startswith(_HEADING_MARK):
                heading = line[len(_HEADING_MARK):].strip()
                continue
            if heading_pattern.match(line):
                heading = as_heading_text(line)
                continue
            result.append((heading, line))

    return result


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last resort for a single 'sentence' longer than max_chars: split on words."""
    parts, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def _chunk_unit(unit: str, max_chars: int) -> list[str]:
    """Keep short units whole; pack sentences of long units into <= max_chars chunks."""
    if len(unit) <= max_chars:
        return [unit]

    sentences = re.split(r"(?<=[.!?])\s+", unit)
    chunks, current = [], ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(sentence, max_chars))
        elif current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _find_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(
            f"Documents location not found: {source}. "
            f"Create the folder and add .txt/.pdf/.docx files, "
            f"or set the DOCUMENTS_DIR environment variable."
        )
    return sorted(
        p for p in source.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.startswith(("~$", "."))  # skip Word lock/temp and hidden files
    )


def load_documents(
    source: Union[str, Path, None] = None,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
    heading_pattern: re.Pattern = DEFAULT_HEADING_PATTERN,
) -> list[dict]:
    """
    Read every supported file under `source` (a folder or a single file)
    and return a list of {"id", "text", "heading", "source", "page"} chunks.
    `text` already has the section heading (if any) prefixed onto it, so
    nothing downstream needs to know sectioning happened; `heading` is kept
    separately too, in case a caller wants it on its own (e.g. for a UI).

    A file that can't be read is logged and skipped, so one bad file
    doesn't stop the whole app from starting.
    """
    source_path = Path(source) if source else DEFAULT_DOCS_DIR
    files = _find_files(source_path)
    if not files:
        logger.warning("No .txt/.pdf/.docx files found in %s", source_path)

    documents: list[dict] = []
    next_id = 1

    for path in files:
        ext = path.suffix.lower()
        try:
            pages = _READERS[ext](path)
        except Exception as e:
            logger.warning("Skipping %s: %s", path.name, e)
            continue

        before = len(documents)
        for page_no, raw_text in pages:
            for heading, unit in _sectionize(raw_text, ext, heading_pattern):
                for chunk in _chunk_unit(unit, max_chunk_chars):
                    if len(chunk) < min_chunk_chars:
                        continue
                    text = f"{heading}. {chunk}" if heading else chunk
                    documents.append({
                        "id": next_id,
                        "text": text,
                        "heading": heading,
                        "source": path.name,
                        "page": page_no,
                    })
                    next_id += 1

        if len(documents) == before:
            hint = " (scanned PDF? it needs OCR)" if ext == ".pdf" else ""
            logger.warning("No text extracted from %s%s", path.name, hint)

    return documents


# Loaded once at import time, so `from app.data import Documents` keeps working.
Documents = load_documents()
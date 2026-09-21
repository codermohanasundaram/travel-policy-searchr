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
 
Output shape (same as the old static list, plus two extra fields):
    [{"id": 1, "text": "...", 
    "source": "policy.pdf", "page": 2}, ...]
 
main.py only uses doc["text"], so nothing else has to change:
    from app.data import Documents
"""
import logging
import os
import re
from pathlib import Path
from typing import Optional,Union

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS={".txt",".pdf",".docx"}
DEFAULT_DOCS_DIR=Path(os.getenv("DOCUMENTS_DIR",Path(__file__).parent/ "documents"))


# Chunking settings.
# Long text is split into chunks of at most MAX_CHUNK_CHARS characters, because
# one embedding per giant document blurs its meaning. Chunks shorter than
# MIN_CHUNK_CHARS (page numbers, stray headings) are dropped as noise.
MAX_CHUNK_CHARS = 500
MIN_CHUNK_CHARS = 20


# ---------------------------------------------------------------------------
# File readers: each returns a list of (page_number_or_None, raw_text) tuples
# ---------------------------------------------------------------------------
 
def _read_txt(path: Path) -> list[tuple[Optional[int], str]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return [(None, text)]
 
 
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
        pages.append((page_no, page.extract_text() or ""))
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
            lines.append(block.text)
    return [(None, "\n".join(lines))]
 
 
_READERS = {
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}



# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
 
def _split_units(text: str, is_pdf: bool) -> list[str]:
    """
    Break raw text into paragraph-sized units.
 
    - txt / docx: every non-empty line is its own paragraph.
    - pdf: lines are hard-wrapped mid-sentence, so paragraphs are separated
      by blank lines and single newlines are joined back into spaces.
    """
    if is_pdf:
        blocks = re.split(r"\n\s*\n", text)
        units = [re.sub(r"\s+", " ", b).strip() for b in blocks]
    else:
        units = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [u for u in units if u]
 
 
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
) -> list[dict]:
    """
    Read every supported file under `source` (a folder or a single file)
    and return a list of {"id", "text", "source", "page"} chunks.
 
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
            for unit in _split_units(raw_text, is_pdf=(ext == ".pdf")):
                for chunk in _chunk_unit(unit, max_chunk_chars):
                    if len(chunk) < min_chunk_chars:
                        continue
                    documents.append({
                        "id": next_id,
                        "text": chunk,
                        "source": path.name,
                        "page": page_no,
                    })
                    next_id += 1
 
        if len(documents) == before:
            hint = " (scanned PDF? it needs OCR)" if ext == ".pdf" else ""
            logger.warning("No text extracted from %s%s", path.name, hint)
 
    return documents
 
 
Documents = load_documents()

# Documents =[
#     {"id":1, "text":"Employees can book economy class for domestic flights."},
#     {"id":2, "text":"Business class is allowed only for international flights longer than 8 hours."},
#     {"id":3, "text":"Hotel accommodation is limited to $200 per night."},
#     {"id":4, "text":"Employees must submit travel expenses within 30 days."},
#     {"id":5, "text":"Employees max taxi claim up to $50 per travel."}
#     ]
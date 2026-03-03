from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
import uuid
import logging

from ..services.data_service import DataService
from ..services.preprocessing_engine import PreprocessingEngine

router = APIRouter()
data_service = DataService()
preprocessing_engine = PreprocessingEngine()
logger = logging.getLogger(__name__)

# ── BOM signatures: detected instantly, zero cost ─────────────────────────────
_BOM_MAP = [
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe",         "utf-16-le"),
    (b"\xfe\xff",         "utf-16-be"),
    (b"\xef\xbb\xbf",    "utf-8-sig"),
]

# ── Priority list — most common first ─────────────────────────────────────────
# latin-1 / cp1252 are placed late because they silently accept any byte;
# we want to try "correct" encodings first so column text is properly decoded.
_ENCODING_FALLBACKS = [
    "utf-8",
    "utf-8-sig",
    "cp1252",      # Windows Western Europe (most common non-UTF8 Windows CSV)
    "cp1250",      # Windows Central Europe
    "cp1251",      # Windows Cyrillic
    "iso-8859-1",  # alias latin-1
    "iso-8859-2",
    "iso-8859-5",
    "iso-8859-9",  # Turkish
    "utf-16",
    "utf-32",
    "gb2312",
    "gb18030",
    "big5",
    "shift_jis",
    "euc-jp",
    "euc-kr",
    "koi8-r",
    "mac-roman",
]


def _bom_encoding(raw: bytes):
    """Return encoding if file starts with a known BOM, else None."""
    for bom, enc in _BOM_MAP:
        if raw.startswith(bom):
            return enc
    return None


def _chardet_encoding(raw: bytes):
    """Sniff encoding from first 64 KB using chardet (if installed)."""
    try:
        import chardet
        result = chardet.detect(raw[:65_536])
        enc  = result.get("encoding")
        conf = result.get("confidence", 0)
        if enc and conf >= 0.70:
            return enc
    except Exception:
        pass
    return None


def _try_read(raw: bytes, encoding: str) -> pd.DataFrame | None:
    """
    Try to read CSV bytes with a given encoding.
    Returns a DataFrame on success, None on ANY failure.
    """
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            encoding=encoding,
            low_memory=False,
            on_bad_lines="skip",   # skip structurally broken lines
        )
        # Sanity check: at least one row and one column
        if df.shape[0] > 0 and df.shape[1] > 0:
            return df
    except Exception as exc:
        logger.debug("Encoding %s failed: %s", encoding, exc)
    return None


def _read_csv_smart(raw: bytes) -> tuple[pd.DataFrame, str]:
    """
    Auto-detect encoding and parse CSV bytes.
    Three-phase strategy (fastest-first):
      1. BOM detection   — O(4 bytes), perfect for BOM files
      2. chardet sniff   — O(64 KB sample), good generic detection
      3. Priority list   — brute-force 19 encodings
      4. Nuclear fallback— latin-1 + errors='replace', can NEVER fail
    """
    tried = set()

    # Phase 1 — BOM
    bom_enc = _bom_encoding(raw)
    if bom_enc:
        df = _try_read(raw, bom_enc)
        if df is not None:
            return df, bom_enc
        tried.add(bom_enc)

    # Phase 2 — chardet
    sniffed = _chardet_encoding(raw)
    if sniffed and sniffed not in tried:
        df = _try_read(raw, sniffed)
        if df is not None:
            return df, sniffed
        tried.add(sniffed)

    # Phase 3 — priority list
    for enc in _ENCODING_FALLBACKS:
        if enc in tried:
            continue
        df = _try_read(raw, enc)
        if df is not None:
            return df, enc
        tried.add(enc)

    # Phase 4 — nuclear fallback: latin-1 + replace; CAN NEVER RAISE
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            encoding="latin-1",
            encoding_errors="replace",
            on_bad_lines="skip",
            low_memory=False,
        )
        return df, "latin-1 (auto-fallback)"
    except Exception as exc:
        # If even this fails the file is structurally broken, not an encoding issue
        raise ValueError(
            f"Could not parse the CSV file even after trying all encodings. "
            f"Ensure the file is a valid CSV. Inner error: {exc}"
        )


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV dataset file.
    Automatically detects encoding (UTF-8, Latin-1, Windows-125x, CJK, Cyrillic, etc.)
    and returns a session ID for subsequent operations.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed. Please upload a .csv file.")

    raw: bytes = b""
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        df, encoding_used = _read_csv_smart(raw)

        # Ensure all column names are plain strings
        df.columns = df.columns.map(str)

        if df.empty:
            raise HTTPException(status_code=400, detail="The CSV file has no data rows.")

        session_id = str(uuid.uuid4())

        data_service.store_data(session_id, df)
        preprocessing_engine.initialize_dataset(session_id, df, file.filename)

        # Sample 7 random rows for preview (or all rows if < 7)
        n_preview = min(5, len(df))
        preview_df = df.sample(n=n_preview, random_state=None).copy()
        preview_df = preview_df.astype(object).where(preview_df.notna(), other=None)
        preview_records = preview_df.to_dict(orient="records")

        logger.info("Uploaded '%s' — encoding=%s, shape=%s", file.filename, encoding_used, df.shape)

        return {
            "session_id":       session_id,
            "message":          "Dataset uploaded successfully",
            "encoding_detected": encoding_used,
            "shape":            list(df.shape),
            "columns":          list(df.columns),
            "preview":          preview_records,
        }

    except HTTPException:
        raise   # pass through our own HTTPExceptions unchanged

    except ValueError as ve:
        # Raised by _read_csv_smart when file is truly unparseable
        logger.error("CSV parse failure for '%s': %s", file.filename, ve)
        raise HTTPException(status_code=422, detail=str(ve))

    except Exception as exc:
        logger.error("Unexpected upload error for '%s': %s", file.filename, exc, exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process file: {str(exc)}"
        )
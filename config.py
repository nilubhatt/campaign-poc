"""Configuration for the campaign-POC service. All knobs are env-overridable."""
import os
from pathlib import Path


def _default_data_dir() -> Path:
    """Platform-appropriate default data location. Windows → %LOCALAPPDATA%, else ~."""
    override = os.getenv("CAMPAIGN_POC_DATA")
    if override:
        return Path(override)
    if os.name == "nt":  # Windows
        base = os.getenv("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "CampaignIntelligence"
    return Path.home() / "campaign-poc-data"


# ── storage ──────────────────────────────────────────────────────────────────
DATA_DIR = _default_data_dir()
DB_PATH = Path(os.getenv("CAMPAIGN_POC_DB", DATA_DIR / "campaigns.db"))
ASSET_DIR = DATA_DIR / "assets"          # stored original decks
UPLOAD_DIR = DATA_DIR / "uploads"        # staging for over-the-wire uploads

# ── file processing ──────────────────────────────────────────────────────────
MAX_DOC_TEXT_CHARS = int(os.getenv("CAMPAIGN_POC_MAX_DOC_CHARS", "40000"))
MAX_PDF_PAGES = int(os.getenv("CAMPAIGN_POC_MAX_PDF_PAGES", "60"))
MAX_PPTX_SLIDES = int(os.getenv("CAMPAIGN_POC_MAX_PPTX_SLIDES", "120"))
# Chunk size for embedding (§6.1): well under the ~6.8k-9k char point where a whole-deck
# embed() call started silently failing against real providers.
MAX_CHUNK_CHARS = int(os.getenv("CAMPAIGN_POC_MAX_CHUNK_CHARS", "1800"))
# Perceptual-hash Hamming distance below which two images count as a match (§6.6). imagehash's
# default phash is 64 bits; 8 tolerates resize/recompress/light crop without matching unrelated images.
PHASH_MATCH_THRESHOLD = int(os.getenv("CAMPAIGN_POC_PHASH_THRESHOLD", "8"))
# POC accepts only presentation/document formats — the use case is decks + briefs.
ALLOWED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.ms-powerpoint",  # .ppt (legacy, stored only)
}
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
PPT_LEGACY_MIME = "application/vnd.ms-powerpoint"
MAX_ASSET_BYTES = int(os.getenv("CAMPAIGN_POC_MAX_ASSET_MB", "100")) * 1024 * 1024
MAX_INLINE_BYTES = int(os.getenv("CAMPAIGN_POC_MAX_INLINE_MB", "10")) * 1024 * 1024

# ── embeddings (semantic search) ─────────────────────────────────────────────
# Anthropic has no embeddings API; a separate embedder powers similarity search.
# Default is Ollama — LOCAL and FREE (run `ollama pull nomic-embed-text` once).
# Voyage is an optional swap (better quality, but a paid API key), never the default.
EMBED_PROVIDER = os.getenv("CAMPAIGN_POC_EMBED_PROVIDER", "ollama").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("CAMPAIGN_POC_OLLAMA_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("CAMPAIGN_POC_EMBED_DIM", "768"))  # nomic-embed-text = 768
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL = os.getenv("CAMPAIGN_POC_VOYAGE_MODEL", "voyage-3.5")

# ── http / auth ──────────────────────────────────────────────────────────────
HTTP_HOST = os.getenv("CAMPAIGN_POC_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("CAMPAIGN_POC_PORT", "8080"))
PUBLIC_BASE_URL = os.getenv("CAMPAIGN_POC_BASE_URL", f"http://localhost:{HTTP_PORT}").rstrip("/")
AUTH_MODE = os.getenv("CAMPAIGN_POC_AUTH_MODE", "none").lower()  # legacy display value
# Pluggable auth: names the registered AuthProvider that handles requests (see auth.py).
# 'none' (default) = no enforcement. Future: 'entra' (Azure AD / Entra ID), 'oidc' (generic),
# 'okta', 'google', … — deploy for a different org by registering a provider + setting this.
AUTH_PROVIDER = os.getenv("CAMPAIGN_POC_AUTH_PROVIDER", AUTH_MODE if AUTH_MODE != "none" else "none").lower()


def ensure_dirs() -> None:
    for d in (DATA_DIR, ASSET_DIR, UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)

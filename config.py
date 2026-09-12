"""Configuration for the campaign-POC service. All knobs are env-overridable."""
import os
import sys
from pathlib import Path


def app_dir() -> Path:
    """The directory the application lives in — where data shipped *with* it is found.

    Frozen (PyInstaller onedir) the layout is <app>/campaign-intelligence[.exe] plus
    <app>/_internal/, so this is the install directory the installers write into Program
    Files. Deliberately NOT sys._MEIPASS (that is _internal/, inside the bundle, where an
    admin can neither see nor replace a file) and NOT __file__ (which in a frozen app points
    into the bundle rather than beside the executable)."""
    if is_installed():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def is_installed() -> bool:
    """True for the packaged binary a customer installed, false for a source checkout.

    The distinction decides whether missing weights are an error: an installed copy ships
    them, so their absence means something went wrong and the product must say so rather
    than quietly reaching for a network that may be blocked. A checkout never had them and
    resolving the tag is the normal developer path."""
    return getattr(sys, "frozen", False)


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

# ── CLIP visual embeddings (§6.6 second half) ─────────────────────────────────
# Aesthetic/regional similarity ("looks like the APAC shoot"), not exact reuse (pHash's job).
# 'hash' is an offline, dependency-free test provider (same role as embedding.py's `hash`
# text provider) - real similarity needs 'openclip' (default), which needs torch and the
# checkpoint below.
CLIP_PROVIDER = os.getenv("CAMPAIGN_POC_CLIP_PROVIDER", "openclip").lower()
# quickgelu variant matches OpenAI's original released weights exactly (open_clip warns on
# an activation-function mismatch otherwise, which would subtly degrade embeddings).
CLIP_MODEL_NAME = os.getenv("CAMPAIGN_POC_CLIP_MODEL", "ViT-B-32-quickgelu")
CLIP_PRETRAINED = os.getenv("CAMPAIGN_POC_CLIP_PRETRAINED", "openai")
# Where the installers put the shipped checkpoint: beside the executable, so a clean
# machine with nothing configured and no network still has working visual search — the
# installer acceptance criterion. Used when no path is configured explicitly. A source
# checkout has no such folder and resolves the tag instead (`python scripts/fetch_weights.py
# models` opts a checkout into the offline behaviour too).
BUNDLED_WEIGHTS_DIR = app_dir() / "models"

# Filesystem path to a CLIP checkpoint (a file, or a directory containing one), used INSTEAD
# of the shipped copy and instead of resolving CLIP_PRETRAINED through the Hugging Face Hub.
# The supported answer for weights supplied out of band, and for an install where
# huggingface.co is blocked by an endpoint filter (hit in the field; the fallback was
# hand-fabricating a HF cache entry). Verified empirically, not assumed: with a local path,
# loading makes zero network calls and produces vectors bit-identical to the tag, while the
# tag path issues live requests even with a warm cache.
# The prefixed name is authoritative, matching every other setting here; the bare name is
# accepted because it is what an admin would guess, and no library reads it.
_CLIP_WEIGHTS_ENV_VARS = ("CAMPAIGN_POC_CLIP_WEIGHTS_PATH", "CLIP_WEIGHTS_PATH")
CLIP_WEIGHTS_PATH = ""
CLIP_WEIGHTS_ENV_VAR = _CLIP_WEIGHTS_ENV_VARS[0]  # which name to name in an error
for _var in _CLIP_WEIGHTS_ENV_VARS:
    if os.getenv(_var):
        CLIP_WEIGHTS_PATH, CLIP_WEIGHTS_ENV_VAR = os.getenv(_var, ""), _var
        break
CLIP_EMBED_DIM = int(os.getenv("CAMPAIGN_POC_CLIP_DIM", "512"))  # ViT-B-32 = 512

# §6.8: find_similar trims each evidence row's freeform `detail` to this many chars by
# default (full campaign briefs can be long; a similarity scan doesn't need all of it up
# front) - pass full_detail=True, or call get_campaign, for the untrimmed record.
EVIDENCE_DETAIL_SUMMARY_CHARS = int(os.getenv("CAMPAIGN_POC_EVIDENCE_DETAIL_CHARS", "300"))
# Same idea, for a match's metrics list (a bulk-imported campaign can carry many rows).
EVIDENCE_METRICS_MAX = int(os.getenv("CAMPAIGN_POC_EVIDENCE_METRICS_MAX", "5"))
# Decks/briefs (extracted for search) plus images (§6.6 — pHash/CLIP; not text-extracted).
ALLOWED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.ms-powerpoint",  # .ppt (legacy, stored only)
}
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
PPT_LEGACY_MIME = "application/vnd.ms-powerpoint"
# review found upload_image_asset's docstring claimed POST /upload works for images while
# ALLOWED_MIME (deck-only) silently rejected them — this is that route's actual allow-list.
ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp"}
# Images embedded inside an uploaded deck, extracted automatically on upload_campaign (via
# asset_ref) so creative-reuse detection doesn't require re-uploading every image by hand -
# "who'll use it then?" Capped to bound per-upload phash/CLIP cost on a deck with many images.
MAX_EXTRACTED_IMAGES_PER_DECK = int(os.getenv("CAMPAIGN_POC_MAX_EXTRACTED_IMAGES", "20"))
MAX_ASSET_BYTES = int(os.getenv("CAMPAIGN_POC_MAX_ASSET_MB", "100")) * 1024 * 1024
MAX_INLINE_BYTES = int(os.getenv("CAMPAIGN_POC_MAX_INLINE_MB", "10")) * 1024 * 1024

# ── time budgets (defect 04: nothing may outlive the transport) ──────────────
# MCP transports drop a tool call that takes too long — the reviewer saw "Device did not
# respond within 60s", with the row already written and no way to tell what had finished.
# A handler must therefore finish, or stop and explain, on its own terms rather than being
# cut off. This is the allowance a handler works within, kept clear of the ceiling below.
def _positive_seconds(var: str, default: str) -> float:
    """Reject a value that would silently disable the protection it configures. "0" stores
    every upload with nothing embedded; "nan" defeats every comparison at once (a deadline
    is never reached AND every grant collapses to the floor) — both fail quietly, which is
    the worst way for a safety limit to fail."""
    raw = os.getenv(var, default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{var}={raw!r} is not a number") from None
    if not value > 0 or value != value:   # value != value catches NaN
        raise ValueError(f"{var}={raw!r} must be a positive number of seconds")
    return value


# Names carry the unit, like every sibling (CAMPAIGN_POC_MAX_ASSET_MB, _MAX_DOC_CHARS).
TOOL_TIME_BUDGET_SECONDS = _positive_seconds("CAMPAIGN_POC_TOOL_BUDGET_SECONDS", "45")
# One embed call, bounded well under that ceiling: a handler embeds once per chunk, so a
# per-call timeout equal to the ceiling (the old value) let a single deck block for minutes.
EMBED_TIMEOUT_SECONDS = _positive_seconds("CAMPAIGN_POC_EMBED_TIMEOUT_SECONDS", "15")
# A health check must answer far faster than the thing it is checking; inheriting the
# timeout it exists to diagnose is how "is it working?" became a 60-second wait.
HEALTH_PROBE_SECONDS = _positive_seconds("CAMPAIGN_POC_HEALTH_PROBE_SECONDS", "3")
# What the transport itself allows. The budget above must stay clear of it; this is the
# number it is clear OF, recorded so the relationship is visible rather than implied.
TRANSPORT_CEILING_SECONDS = _positive_seconds("CAMPAIGN_POC_TRANSPORT_CEILING_SECONDS", "60")

# ── embeddings (semantic search) ─────────────────────────────────────────────
# Anthropic has no embeddings API; a separate embedder powers similarity search.
# Default is Ollama — LOCAL and FREE (run `ollama pull nomic-embed-text` once).
# Voyage is an optional swap (better quality, but a paid API key), never the default.
EMBED_PROVIDER = os.getenv("CAMPAIGN_POC_EMBED_PROVIDER", "ollama").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("CAMPAIGN_POC_OLLAMA_MODEL", "nomic-embed-text")
# How long Ollama should keep the embedding model resident. Default "-1" = indefinitely,
# which trades ~270MB of the user's RAM for never paying a model reload inside a tool call —
# precisely the first-use cost that blows a transport ceiling. Set a duration ("5m") on a
# memory-constrained machine.
def _keep_alive(raw: str):
    """Ollama accepts a NUMBER of seconds (-1 = keep indefinitely) or a duration string with
    a unit ("10m"). It rejects a numeric string: `keep_alive: "-1"` fails with
    `time: missing unit in duration "-1"`, which took every embed call down with a 400."""
    try:
        return int(raw)
    except ValueError:
        return raw   # a duration like "10m", passed through as Ollama expects


OLLAMA_KEEP_ALIVE = _keep_alive(os.getenv("CAMPAIGN_POC_OLLAMA_KEEP_ALIVE", "-1"))
EMBED_DIM = int(os.getenv("CAMPAIGN_POC_EMBED_DIM", "768"))  # nomic-embed-text = 768
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL = os.getenv("CAMPAIGN_POC_VOYAGE_MODEL", "voyage-3.5")

# ── http / auth ──────────────────────────────────────────────────────────────
HTTP_HOST = os.getenv("CAMPAIGN_POC_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("CAMPAIGN_POC_PORT", "8086"))
PUBLIC_BASE_URL = os.getenv("CAMPAIGN_POC_BASE_URL", f"http://localhost:{HTTP_PORT}").rstrip("/")
AUTH_MODE = os.getenv("CAMPAIGN_POC_AUTH_MODE", "none").lower()  # legacy display value
# Pluggable auth: names the registered AuthProvider that handles requests (see auth.py).
# 'none' (default) = no enforcement. Future: 'entra' (Azure AD / Entra ID), 'oidc' (generic),
# 'okta', 'google', … — deploy for a different org by registering a provider + setting this.
AUTH_PROVIDER = os.getenv("CAMPAIGN_POC_AUTH_PROVIDER", AUTH_MODE if AUTH_MODE != "none" else "none").lower()


def ensure_dirs() -> None:
    for d in (DATA_DIR, ASSET_DIR, UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)

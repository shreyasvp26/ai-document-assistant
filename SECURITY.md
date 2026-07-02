# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `main`  | ✅        |
| Older versions | ❌ |

## Security Measures Implemented

### API Key Management
- **No hardcoded keys** — all secrets loaded from environment variables via `python-dotenv`
- **Startup validation** — app fails fast with a clear message if required env vars (`GOOGLE_API_KEY`) are missing
- **`.env` gitignored** — `.env.example` committed with placeholder values only
- **Deployment secrets** — stored in Streamlit Community Cloud's secrets manager, never in the repo

### Prompt Injection Defense
- **Input sanitization** (`src/safety.py`) — detects and neutralizes known prompt injection patterns in user input:
  - "Ignore previous instructions" variants
  - "System:" / "ADMIN:" prefix attacks
  - Role-playing injection attempts
  - Encoded/obfuscated injection attempts
- **Document content sanitization** — retrieved document chunks are also scanned for injection patterns before being sent to the model
- **Grounded prompting** — system prompts explicitly instruct the model to only answer from provided context

### PII Protection
- **Output redaction** (`src/safety.py`) — regex-based masking of:
  - Email addresses: `user@example.com` → `u***@***.com`
  - Phone numbers: `+1-234-567-8901` → `+1-***-***-8901`
- **Applied automatically** — all model outputs pass through redaction before reaching the user

### Input Validation
- **File type validation** — only PDF uploads accepted
- **File size limits** — enforced at the Streamlit layer
- **Empty file detection** — graceful error for 0-byte or unparseable PDFs
- **Calculator safety** — arithmetic expressions parsed via `ast` module with a strict whitelist of allowed node types; no `eval()` on raw strings

### Error Handling
- **Every external call** (Gemini API, Tavily API, file I/O, ChromaDB) wrapped in try/except
- **User-facing error messages** — no raw stack traces shown in the Streamlit UI
- **Graceful degradation** — missing Tavily key disables web search but doesn't crash the app

### Secret Scanning
- **Pre-commit hooks** — detect hardcoded secrets before they reach the repo
- **GitHub Actions** — Gitleaks workflow scans every push and PR
- **Gitleaks config** — custom rules for Python-specific secret patterns

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, open a GitHub issue **only if** the report contains **no sensitive details**, or contact the maintainer directly.

### What to include
- Description of the vulnerability and potential impact
- Steps to reproduce
- Affected files/versions
- Any known mitigations

### Response timeline
- Acknowledge within **72 hours**
- Status update within **7 days**

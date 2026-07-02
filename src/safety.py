"""Safety module — input sanitization and output PII redaction.

This module provides two critical safety layers:
1. Input sanitization: detects and neutralizes prompt injection attempts
   in both user input and retrieved document content
2. Output redaction: regex-masks emails and phone numbers in generated responses

Design decisions:
- Neutralize injections rather than blocking entirely — strips the malicious pattern
  but still processes the remaining content
- PII redaction uses regex patterns that cover common email and phone formats
- All detections are logged for monitoring but not exposed to the user
- Patterns are documented inline for interviewer readability
"""

import re
import logging

logger = logging.getLogger(__name__)


# --- Prompt Injection Detection ---

# Patterns that indicate prompt injection attempts.
# Each tuple: (compiled regex, description for logging)
# These cover the most common injection vectors while minimizing false positives.
INJECTION_PATTERNS = [
    # "Ignore previous instructions" and variants
    (
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier|preceding)\s+"
            r"(instructions?|prompts?|rules?|context|directives?)",
            re.IGNORECASE,
        ),
        "ignore-previous-instructions",
    ),
    # "Forget everything" variants
    (
        re.compile(
            r"forget\s+(everything|all|what)\s+(you|I|we)\s+"
            r"(said|told|discussed|know)",
            re.IGNORECASE,
        ),
        "forget-everything",
    ),
    # System/admin role injection: "System: ..." or "ADMIN: ..."
    (
        re.compile(
            r"^(system|admin|administrator|root|supervisor)\s*:\s*",
            re.IGNORECASE | re.MULTILINE,
        ),
        "role-prefix-injection",
    ),
    # "You are now..." role reassignment
    (
        re.compile(
            r"you\s+are\s+now\s+(a|an|the|my)\s+",
            re.IGNORECASE,
        ),
        "role-reassignment",
    ),
    # "Act as..." or "Pretend to be..."
    (
        re.compile(
            r"(act|pretend|behave|respond)\s+(as|like)\s+(a|an|the|if)\s+",
            re.IGNORECASE,
        ),
        "role-playing",
    ),
    # "New instructions:" or "Updated prompt:"
    (
        re.compile(
            r"(new|updated|revised|override|replacement)\s+"
            r"(instructions?|prompts?|rules?|system\s*message)",
            re.IGNORECASE,
        ),
        "instruction-override",
    ),
    # Delimiter injection — trying to break out of the prompt structure
    (
        re.compile(
            r"```\s*(system|instruction|prompt|admin)",
            re.IGNORECASE,
        ),
        "delimiter-injection",
    ),
    # "Do not follow" / "disregard" instructions
    (
        re.compile(
            r"(do\s+not|don'?t|never)\s+(follow|obey|listen|adhere)\s+"
            r"(to\s+)?(the\s+)?(previous|above|system|original)",
            re.IGNORECASE,
        ),
        "instruction-negation",
    ),
    # "Reveal your prompt" / "Show me your instructions"
    (
        re.compile(
            r"(reveal|show|display|print|output|tell\s+me)\s+"
            r"(your|the|system)\s+(prompt|instructions?|rules?|system\s*message)",
            re.IGNORECASE,
        ),
        "prompt-extraction",
    ),
]


def detect_injection(text: str) -> list[str]:
    """Scan text for prompt injection patterns.

    Args:
        text: The text to scan (user input or document content).

    Returns:
        List of detected injection pattern names (empty if clean).
    """
    detections = []
    for pattern, name in INJECTION_PATTERNS:
        if pattern.search(text):
            detections.append(name)
    return detections


def sanitize_input(text: str) -> str:
    """Sanitize user input by neutralizing detected prompt injection patterns.

    Rather than blocking the entire input, this function strips the
    injection patterns while preserving the rest of the text. This
    approach is more user-friendly — a partial injection in an otherwise
    legitimate query still gets processed.

    Args:
        text: The user's input text.

    Returns:
        Sanitized text with injection patterns removed.
    """
    if not text:
        return text

    detections = detect_injection(text)
    if not detections:
        return text

    # Log the detection for monitoring (but don't expose to user)
    logger.warning(
        f"Prompt injection detected in user input: {detections}. "
        f"Input (truncated): '{text[:100]}...'"
    )

    sanitized = text
    for pattern, name in INJECTION_PATTERNS:
        sanitized = pattern.sub("", sanitized)

    # Clean up any leftover whitespace from removals
    sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()

    return sanitized


def sanitize_document_content(text: str) -> str:
    """Sanitize retrieved document content before it reaches the model.

    Documents could contain adversarial text planted to manipulate the LLM.
    This function applies the same injection detection to document chunks.

    Args:
        text: The document chunk text.

    Returns:
        Sanitized document text.
    """
    if not text:
        return text

    detections = detect_injection(text)
    if not detections:
        return text

    logger.warning(
        f"Prompt injection detected in document content: {detections}. "
        f"Content (truncated): '{text[:100]}...'"
    )

    sanitized = text
    for pattern, name in INJECTION_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)

    return sanitized


# --- PII Redaction ---

# Email pattern: matches common email formats
# e.g., user@example.com → u***@***.com
EMAIL_PATTERN = re.compile(
    r"\b([a-zA-Z0-9._%+-])([a-zA-Z0-9._%+-]*)@"
    r"([a-zA-Z0-9.-])([a-zA-Z0-9.-]*)\.([a-zA-Z]{2,})\b"
)

# Phone patterns: covers multiple formats
# US: (123) 456-7890, 123-456-7890, +1-234-567-8901
# International: +91-1234567890, +44 20 7123 4567
PHONE_PATTERNS = [
    # US format with parentheses: (123) 456-7890
    re.compile(
        r"\((\d{3})\)\s*(\d{3})[-.](\d{4})"
    ),
    # US/International with country code: +1-234-567-8901 or +91-1234567890
    re.compile(
        r"(\+\d{1,3})[-.\s]?(\d{1,4})[-.\s]?(\d{3,4})[-.\s]?(\d{3,4})"
    ),
    # Simple US format: 123-456-7890 or 123.456.7890
    re.compile(
        r"\b(\d{3})[-.](\d{3})[-.](\d{4})\b"
    ),
    # 10-digit number: 1234567890
    re.compile(
        r"\b(\d{10})\b"
    ),
]


def redact_email(text: str) -> str:
    """Redact email addresses in the given text.

    Preserves the first character and domain TLD for readability
    while masking the identifying parts.

    Args:
        text: Text that may contain email addresses.

    Returns:
        Text with email addresses redacted.

    Examples:
        >>> redact_email("Contact john.doe@example.com")
        'Contact j***@***.com'
    """
    def _mask_email(match):
        first_char = match.group(1)
        tld = match.group(5)
        return f"{first_char}***@***.{tld}"

    return EMAIL_PATTERN.sub(_mask_email, text)


def redact_phone(text: str) -> str:
    """Redact phone numbers in the given text.

    Preserves the format structure but masks the digits.

    Args:
        text: Text that may contain phone numbers.

    Returns:
        Text with phone numbers redacted.

    Examples:
        >>> redact_phone("Call (123) 456-7890")
        'Call (***) ***-7890'
        >>> redact_phone("Call +1-234-567-8901")
        'Call +1-***-***-8901'
    """
    result = text

    # Pattern 1: (123) 456-7890 → (***) ***-7890
    result = PHONE_PATTERNS[0].sub(
        lambda m: f"(***) ***-{m.group(3)}", result
    )

    # Pattern 2: +1-234-567-8901 → +1-***-***-8901
    result = PHONE_PATTERNS[1].sub(
        lambda m: f"{m.group(1)}-***-***-{m.group(4)}", result
    )

    # Pattern 3: 123-456-7890 → ***-***-7890
    result = PHONE_PATTERNS[2].sub(
        lambda m: f"***-***-{m.group(3)}", result
    )

    # Pattern 4: 1234567890 → ******7890
    result = PHONE_PATTERNS[3].sub(
        lambda m: f"******{m.group(1)[-4:]}", result
    )

    return result


def redact_pii(text: str) -> str:
    """Redact all PII (emails and phone numbers) from the given text.

    This is the main PII redaction function — call this on all model
    outputs before showing them to the user.

    Args:
        text: The model's response text.

    Returns:
        Text with all detected PII redacted.
    """
    if not text:
        return text

    result = redact_email(text)
    result = redact_phone(result)
    return result


def apply_safety(user_input: str) -> str:
    """Apply full input safety pipeline to user input.

    Combines injection detection and sanitization into a single call.

    Args:
        user_input: Raw user input text.

    Returns:
        Sanitized user input safe for model processing.
    """
    return sanitize_input(user_input)


def apply_output_safety(model_output: str) -> str:
    """Apply full output safety pipeline to model output.

    Redacts PII from the model's response before it reaches the user.

    Args:
        model_output: Raw model response text.

    Returns:
        Response text with PII redacted.
    """
    return redact_pii(model_output)

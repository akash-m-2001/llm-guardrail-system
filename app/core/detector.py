"""Rule-based detection.

Patterns match whole words and phrases, not substrings: "hackathon" must not
look like "hack", and "don't ignore the safety rules" must not look like an
instruction-override attack. Each pattern carries its own weight, so a strong
signal (asking for the system prompt) scores higher than a weak one.
"""
import re

_I = re.IGNORECASE

# Not preceded by a negation: "don't ignore the rules" is advice, not an attack.
_NOT_NEGATED = r"(?<!\bdon't )(?<!\bdont )(?<!\bdo not )(?<!\bnever )(?<!\bnot )"

INJECTION_PATTERNS = [
    # "ignore / disregard / forget ... previous / above / all ... instructions"
    ("override_instructions", 0.8, re.compile(
        _NOT_NEGATED + r"\b(ignore|disregard|forget|override|bypass)\b[\w\s,'-]{0,30}?"
        r"\b(previous|prior|above|earlier|preceding|all|your|system)\b[\w\s,'-]{0,20}?"
        r"\b(instructions?|rules|prompts?|directions|guidelines|messages?|context)\b", _I)),
    ("forget_everything", 0.7, re.compile(
        _NOT_NEGATED + r"\b(forget|ignore|disregard)\s+(everything|all)\b(\s+(above|before|so far|previous(ly)?))?", _I)),
    ("reveal_system_prompt", 0.8, re.compile(
        r"\b(reveal|print|show|repeat|output|tell me|display|leak)\b[\w\s,'-]{0,30}?\b(system|hidden|initial|original)\s+(prompt|instructions?|message)\b", _I)),
    ("role_override", 0.6, re.compile(r"\byou\s+are\s+(now|no\s+longer)\b|\bfrom\s+now\s+on\s+you\b|\bpretend\s+(you\s+are|to\s+be)\b", _I)),
    ("jailbreak_persona", 0.8, re.compile(r"\b(DAN|do\s+anything\s+now|developer\s+mode|jailbreak(ed)?|unrestricted\s+mode)\b", _I)),
    ("fake_role_marker", 0.5, re.compile(r"(^|\n)\s*(system|assistant|developer)\s*:", _I)),
]

HARMFUL_PATTERNS = [
    ("hacking", 0.7, re.compile(r"\bhack(ing|ed)?\s+(into|a|an|the|someone'?s?|my\s+(ex|neighbou?r|boss))\b", _I)),
    ("malware", 0.8, re.compile(r"\b(malware|ransomware|keylogger|trojan|botnet|rootkit|spyware)\b", _I)),
    ("credential_theft", 0.7, re.compile(r"\b(steal|phish(ing)?|crack|harvest)\b[\w\s'-]{0,20}?\b(passwords?|credentials|logins?|credit\s+cards?)\b", _I)),
    ("weapons", 0.9, re.compile(r"\b(make|build|assemble|synthesi[sz]e)\b[\w\s'-]{0,20}?\b(bomb|explosives?|pipe\s+bomb|nerve\s+agent|bioweapon)\b", _I)),
]


def _match(patterns, text):
    hits = [(name, weight) for name, weight, pattern in patterns if pattern.search(text)]
    return [name for name, _ in hits], max((w for _, w in hits), default=0.0)


def detect_risk(prompt: str) -> dict:
    injection_hits, injection_score = _match(INJECTION_PATTERNS, prompt)
    harmful_hits, harmful_score = _match(HARMFUL_PATTERNS, prompt)

    categories = []
    if injection_hits:
        categories.append("prompt_injection")
    if harmful_hits:
        categories.append("harmful_content")

    return {
        "categories": categories,
        "keyword_hits": injection_hits + harmful_hits,
        "prompt_risk_score": max(injection_score, harmful_score),
    }

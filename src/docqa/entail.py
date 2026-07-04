"""Entailment gate (R-ENTAIL) — the inductive-step check: does the cited span actually SUPPORT
the claim, not merely resolve to a real record?

Referential integrity (BT13) proves a citation points at a real retrieved record. This gate proves
the record's text ENTAILS the answer's proposition. Without it, a claim can cite a real-but-
irrelevant span and pass (the gap the BT13 tests documented). Support-checking is genuine NLP, so
per the operating model we hand it to the agent (an LLM entailment judge, temp 0), gated
deterministically: the pipeline decides drop-or-keep on a binary verdict, and the eval asserts the
observable outcome (unentailed claim dropped -> refuse), robust to wording.

The judge is injectable so tests drive it with a stub (no API); the Anthropic judge is lazy.
"""

from __future__ import annotations


class EntailmentJudge:
    """Callable: (claim_text, span_text) -> bool (does span entail claim?). Base = permissive
    stub; the real judge subclasses / is injected."""

    def __call__(self, claim_text: str, span_text: str) -> bool:
        return True


class AnthropicEntailmentJudge:
    """LLM entailment judge. One temp=0 call per (claim, span) pair; binary verdict parsed from a
    reserved token. Lazy import; only the answer path needs a key."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def __call__(self, claim_text: str, span_text: str) -> bool:
        from anthropic import Anthropic

        from docqa.config import require_api_key

        system = (
            "You are a strict entailment judge. Given a SPAN of source text and a CLAIM, decide "
            "whether the SPAN alone logically supports (entails) the CLAIM. Answer with ONLY one "
            "word: ENTAILED if the span supports the claim, or NOT_ENTAILED if it does not, is "
            "unrelated, or only partially supports it. Do not explain."
        )
        user = f"SPAN:\n{span_text}\n\nCLAIM:\n{claim_text}"
        client = Anthropic(api_key=require_api_key())
        msg = client.messages.create(
            model=self.model_id, max_tokens=8, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content).strip().upper()
        # Conservative: only an explicit ENTAILED (and not NOT_ENTAILED) passes.
        return "ENTAILED" in text and "NOT_ENTAILED" not in text


def batch_entails(judge, pairs: list[tuple[str, str]]) -> list[bool]:
    """Run the judge over (claim, span) pairs. Kept simple; a real judge may batch internally."""
    return [judge(claim, span) for claim, span in pairs]

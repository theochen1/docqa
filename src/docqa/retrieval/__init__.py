"""Retrieval — pluggable implementations behind the Retriever seam.

Everything here returns claims ordered by score with a stable, deterministic tie-break. No
implementation decides correctness or conflict; retrieval SELECTS, the verifier DISPOSES.
"""

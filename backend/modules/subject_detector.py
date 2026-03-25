"""
modules/subject_detector.py
============================
Detects the academic subject of a page from its OCR text.

Method
------
1. Keyword scoring — weighted vocabulary per subject
2. Formula density — maths symbols → Math/Physics/Chemistry
3. Returns best match with confidence score

Supports:
  Mathematics, Physics, Chemistry, Biology, History,
  Geography, Computer Science, Economics, Literature,
  Law, General / Unknown
"""

from __future__ import annotations

import re
from collections import defaultdict


# ── Subject keyword vocabularies ─────────────────────────────────────────────
# Format: {subject_name: [(keyword/phrase, weight), ...]}

SUBJECT_VOCAB: dict[str, list[tuple[str, float]]] = {

    "Mathematics": [
        ("equation", 2.0), ("polynomial", 3.0), ("derivative", 3.0),
        ("integral", 3.0), ("matrix", 2.5), ("determinant", 3.0),
        ("eigenvalue", 4.0), ("vector", 2.0), ("scalar", 2.0),
        ("theorem", 1.5), ("proof", 1.5), ("lemma", 3.0),
        ("corollary", 3.0), ("hypothesis", 1.0), ("trigonometry", 3.0),
        ("calculus", 3.0), ("algebra", 2.5), ("geometry", 2.5),
        ("probability", 2.0), ("statistics", 2.0), ("logarithm", 3.0),
        ("factorial", 3.0), ("prime number", 2.5), ("modulo", 2.5),
        ("function", 1.5), ("limit", 1.5), ("continuity", 2.0),
        ("differentiation", 3.0), ("integration", 3.0),
        ("coordinate", 1.5), ("parabola", 2.5), ("hyperbola", 2.5),
        ("arithmetic", 2.0), ("sequence", 1.5), ("series", 1.5),
        ("permutation", 2.5), ("combination", 2.0), ("binomial", 2.5),
    ],

    "Physics": [
        ("velocity", 3.0), ("acceleration", 3.0), ("momentum", 3.0),
        ("force", 2.0), ("energy", 1.5), ("kinetic", 2.5), ("potential", 1.5),
        ("newton", 2.5), ("gravity", 2.5), ("friction", 2.5),
        ("thermodynamics", 4.0), ("entropy", 3.5), ("enthalpy", 3.5),
        ("quantum", 4.0), ("photon", 3.5), ("wavelength", 3.0),
        ("frequency", 2.0), ("amplitude", 2.5), ("oscillation", 3.0),
        ("electric", 2.0), ("magnetic", 2.0), ("electromagnetic", 3.5),
        ("resistance", 2.0), ("current", 1.5), ("voltage", 2.5),
        ("refraction", 3.0), ("reflection", 2.0), ("diffraction", 3.0),
        ("capacitor", 3.0), ("inductor", 3.0), ("circuit", 2.0),
        ("nuclear", 3.5), ("relativity", 4.0), ("optics", 3.0),
        ("pressure", 1.5), ("density", 1.5), ("buoyancy", 3.0),
        ("torque", 3.0), ("angular momentum", 3.5),
    ],

    "Chemistry": [
        ("atom", 2.0), ("molecule", 2.5), ("element", 1.5), ("compound", 2.0),
        ("reaction", 2.0), ("bond", 1.5), ("covalent", 3.0), ("ionic", 3.0),
        ("oxidation", 3.0), ("reduction", 2.5), ("valence", 3.0),
        ("electron", 2.0), ("proton", 2.0), ("neutron", 2.0),
        ("periodic table", 4.0), ("mole", 2.5), ("molarity", 3.5),
        ("acid", 2.0), ("base", 1.5), ("pH", 3.0), ("titration", 4.0),
        ("catalyst", 3.0), ("equilibrium", 2.5), ("entropy", 2.0),
        ("enthalpy", 3.0), ("activation energy", 4.0),
        ("isomer", 3.5), ("polymer", 3.0), ("organic", 2.5),
        ("hydrocarbon", 3.5), ("functional group", 4.0),
        ("electrolysis", 4.0), ("precipitation", 3.0),
        ("stoichiometry", 4.0), ("yield", 2.0),
        ("spectroscopy", 4.0), ("chromatography", 4.0),
    ],

    "Biology": [
        ("cell", 1.5), ("nucleus", 2.0), ("membrane", 2.5), ("organelle", 3.0),
        ("DNA", 3.5), ("RNA", 3.5), ("gene", 2.5), ("chromosome", 3.0),
        ("protein", 2.0), ("enzyme", 3.0), ("metabolism", 3.0),
        ("photosynthesis", 4.0), ("respiration", 2.5), ("mitosis", 4.0),
        ("meiosis", 4.0), ("evolution", 3.0), ("natural selection", 4.0),
        ("ecosystem", 3.0), ("biodiversity", 3.5), ("taxonomy", 3.5),
        ("anatomy", 3.0), ("physiology", 3.5), ("hormone", 3.0),
        ("antibody", 3.5), ("immune", 2.5), ("pathogen", 3.5),
        ("neuron", 3.5), ("synapse", 3.5), ("nervous system", 3.5),
        ("heart", 2.0), ("blood", 1.5), ("tissue", 2.0),
        ("species", 2.0), ("habitat", 2.5), ("food chain", 3.0),
        ("chlorophyll", 3.5), ("osmosis", 3.5), ("diffusion", 2.5),
    ],

    "History": [
        ("war", 1.5), ("empire", 2.5), ("revolution", 2.5), ("dynasty", 3.0),
        ("treaty", 3.0), ("colonisation", 3.5), ("independence", 2.5),
        ("century", 2.0), ("civilisation", 3.0), ("ancient", 2.5),
        ("medieval", 3.5), ("renaissance", 3.5), ("industrial", 2.5),
        ("battle", 2.5), ("invasion", 2.5), ("constitution", 2.5),
        ("parliament", 2.5), ("monarchy", 3.0), ("republic", 2.5),
        ("nationalism", 3.0), ("imperialism", 3.5), ("socialism", 3.0),
        ("democracy", 2.0), ("election", 2.0), ("president", 2.0),
        ("world war", 4.0), ("cold war", 4.0),
    ],

    "Geography": [
        ("latitude", 3.5), ("longitude", 3.5), ("altitude", 3.0),
        ("continent", 2.5), ("ocean", 2.0), ("river", 2.0), ("mountain", 2.0),
        ("climate", 2.5), ("precipitation", 2.5), ("erosion", 3.5),
        ("tectonic", 4.0), ("earthquake", 3.0), ("volcano", 3.5),
        ("population", 2.0), ("urbanisation", 3.5), ("migration", 2.5),
        ("resource", 1.5), ("agriculture", 2.5), ("irrigation", 3.0),
        ("biome", 3.5), ("desert", 2.0), ("tropical", 2.5),
        ("monsoon", 3.5), ("glacier", 3.0), ("delta", 2.5),
    ],

    "Computer Science": [
        ("algorithm", 3.0), ("data structure", 4.0), ("array", 2.5),
        ("linked list", 4.0), ("stack", 2.5), ("queue", 2.5),
        ("tree", 2.0), ("graph", 2.0), ("sorting", 2.5), ("searching", 2.5),
        ("recursion", 3.5), ("complexity", 2.5), ("big o", 3.5),
        ("program", 1.5), ("function", 1.5), ("variable", 1.5),
        ("loop", 2.0), ("condition", 1.5), ("class", 2.0), ("object", 2.0),
        ("inheritance", 3.0), ("polymorphism", 3.5), ("database", 2.5),
        ("query", 2.5), ("network", 2.0), ("protocol", 2.5),
        ("operating system", 3.5), ("memory", 2.0), ("cpu", 3.0),
        ("compiler", 3.5), ("runtime", 3.0), ("binary", 2.5),
        ("encryption", 3.5), ("machine learning", 4.0),
    ],

    "Economics": [
        ("supply", 2.5), ("demand", 2.5), ("market", 2.0), ("price", 1.5),
        ("inflation", 3.0), ("gdp", 3.5), ("growth", 1.5), ("recession", 3.5),
        ("fiscal", 3.5), ("monetary", 3.0), ("policy", 1.5),
        ("elasticity", 3.5), ("utility", 2.5), ("marginal", 3.0),
        ("opportunity cost", 4.0), ("trade", 2.0), ("export", 2.5),
        ("import", 2.0), ("tariff", 3.5), ("investment", 2.0),
        ("interest rate", 3.5), ("bond", 2.0), ("stock", 2.0),
        ("monopoly", 3.5), ("oligopoly", 3.5), ("equilibrium", 3.0),
    ],

    "Literature": [
        ("poem", 2.5), ("poetry", 3.0), ("stanza", 3.5), ("metaphor", 3.5),
        ("simile", 3.5), ("imagery", 3.0), ("symbolism", 3.0),
        ("character", 2.0), ("plot", 2.5), ("theme", 2.0), ("setting", 2.0),
        ("narrator", 3.0), ("dialogue", 2.5), ("soliloquy", 4.0),
        ("tragedy", 2.5), ("comedy", 2.0), ("novel", 2.5),
        ("rhyme", 3.0), ("rhythm", 2.5), ("alliteration", 3.5),
        ("protagonist", 3.5), ("antagonist", 3.5), ("irony", 3.0),
        ("foreshadowing", 4.0), ("flashback", 3.5), ("climax", 2.5),
    ],
}

# Math symbols as strong subject indicators
MATH_SYMBOLS = ["∫", "∑", "∏", "√", "∂", "∇", "≈", "≠", "≤", "≥",
                "α", "β", "γ", "δ", "θ", "λ", "μ", "π", "σ", "φ", "ω",
                "²", "³", "⁴", "⁵", "₀", "₁", "₂", "₃"]

CHEM_INDICATORS = re.compile(
    r"\b(H2O|CO2|NaCl|H2SO4|HCl|NaOH|NH3|C6H12O6|CaCO3|Fe2O3|"
    r"[A-Z][a-z]?\d*(?:\+|-|\d)*)\b"
)


class SubjectDetector:

    def detect(self, text: str) -> str:
        """
        Return the most likely subject as a string.
        Returns "General" if no subject is confidently detected.
        """
        if not text or len(text.strip()) < 20:
            return "General"

        text_lower = text.lower()
        scores: dict[str, float] = defaultdict(float)

        # ── Keyword scoring ───────────────────────────────────────────────
        for subject, vocab in SUBJECT_VOCAB.items():
            for keyword, weight in vocab:
                # Count occurrences (case-insensitive, whole word)
                pattern = rf"\b{re.escape(keyword)}\b"
                count   = len(re.findall(pattern, text_lower))
                scores[subject] += count * weight

        # ── Math symbol density boost ────────────────────────────────────
        sym_count = sum(text.count(s) for s in MATH_SYMBOLS)
        if sym_count >= 3:
            scores["Mathematics"] += sym_count * 2.0
            scores["Physics"]     += sym_count * 1.0
            scores["Chemistry"]   += sym_count * 0.5

        # ── Chemical formula detection ───────────────────────────────────
        chem_matches = len(CHEM_INDICATORS.findall(text))
        if chem_matches >= 2:
            scores["Chemistry"] += chem_matches * 3.0

        # ── Equation lines (lines containing = with numbers) ─────────────
        eq_lines = len(re.findall(r"[\w\s]+\s*=\s*[\d\w\s\+\-\*/]+", text))
        if eq_lines >= 2:
            scores["Mathematics"] += eq_lines * 1.5
            scores["Physics"]     += eq_lines * 1.0

        if not scores:
            return "General"

        best_subject = max(scores, key=lambda s: scores[s])
        best_score   = scores[best_subject]

        # Require minimum score of 3 to claim a subject
        if best_score < 3.0:
            return "General"

        return best_subject

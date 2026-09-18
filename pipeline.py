"""
pipeline.py — YourWords four-stage writing-assistance pipeline.

Stages:
    1. Spell check   — list misspellings (each verified against the input)
    2. Spell fix     — apply only the spelling corrections
    3. Grammar check — list grammar issues, grounded in retrieved style rules (RAG)
    4. Grammar fix   — apply grammar corrections, preserving the author's voice

The pipeline is model-agnostic: it depends only on a generate_response(messages)
function (see llm.py) and an optional retriever exposing .retrieve(query, k)
(see retriever.py). Both are injected, so swapping providers or testing with
fakes is trivial.
"""
from __future__ import annotations

import re
from typing import Callable, List

from llm import generate_response as default_llm

LLMFn = Callable[[list], str]


class LLMAgent:
    """A single-purpose agent: a system role + one user message per call."""

    def __init__(self, role_description: str, llm_fn: LLMFn = default_llm):
        self.role_description = role_description
        self.llm_fn = llm_fn

    def inference(self, message: str) -> str:
        messages = [
            {"role": "system", "content": self.role_description},
            {"role": "user", "content": message},
        ]
        return self.llm_fn(messages)


class WritingAssistancePipeline:
    def __init__(self, retriever=None, llm_fn: LLMFn = default_llm, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

        self.spell_check_agent = LLMAgent(
            "You are a contextual spelling specialist that identifies and flags "
            "misspelled words in context.",
            llm_fn,
        )
        self.spell_correction_agent = LLMAgent(
            "You are an editor that fixes only spelling errors while keeping "
            "everything else exactly the same.",
            llm_fn,
        )
        self.grammar_check_agent = LLMAgent(
            "You are an American English grammar expert who finds grammar issues "
            "while preserving the author's original voice.",
            llm_fn,
        )
        self.final_correction_agent = LLMAgent(
            "You are an editor that applies grammar corrections while preserving "
            "the author's original voice and style.",
            llm_fn,
        )

    def process_text(self, raw_text: str) -> dict:
        spelling_issues = self._run_spell_check(raw_text)
        spell_corrected = self._apply_spell_corrections(raw_text, spelling_issues)
        retrieved_rules = self._retrieve_rules(spell_corrected)
        grammar_suggestions = self._run_grammar_check(spell_corrected, retrieved_rules)
        final_text = self._apply_grammar_corrections(spell_corrected, grammar_suggestions)
        return {
            "original_text": raw_text,
            "spelling_issues": spelling_issues,
            "spell_corrected_text": spell_corrected,
            "retrieved_rules": retrieved_rules,
            "grammar_suggestions": grammar_suggestions,
            "final_text": final_text,
        }

    # ---- Stage 1: spell check -------------------------------------------------
    def _run_spell_check(self, text: str) -> str:
        prompt = (
            "Find spelling errors in the text below. Check for missing apostrophes "
            "in contractions, transposed or missing letters, and commonly confused "
            "spellings.\n\n"
            "List each misspelling on its own line, exactly as:\n"
            "- wrong -> correct\n\n"
            "If there are none, reply exactly: No spelling errors found.\n\n"
            f"TEXT:\n{text}"
        )
        raw = self.spell_check_agent.inference(prompt)
        return self._clean_spelling_list(raw, text)

    def _clean_spelling_list(self, result: str, text: str) -> str:
        """
        Parse the model's list into a clean, de-duplicated set of corrections, and
        drop any 'misspelling' whose flagged word does not actually appear in the
        input. This verification step is the pipeline's guard against hallucinated
        corrections.
        """
        if "No spelling errors found" in result:
            return "No spelling errors found."

        text_words = set(re.findall(r"\b[\w']+\b", text.lower()))
        pair = re.compile(
            r"([A-Za-z]+(?:'[A-Za-z]+)?)\s*[-–—>→]+\s*([A-Za-z]+(?:'[A-Za-z]+)?)"
        )

        cleaned, seen = [], set()
        for line in result.splitlines():
            match = pair.search(line.strip())
            if not match:
                continue
            wrong, correct = match.group(1).strip(), match.group(2).strip()
            if wrong.lower() not in text_words:  # hallucination guard
                continue
            key = (wrong.lower(), correct.lower())
            if key in seen or wrong.lower() == correct.lower():
                continue
            seen.add(key)
            cleaned.append(f"- {wrong} → {correct}")

        return "\n".join(cleaned) if cleaned else "No spelling errors found."

    def _apply_spell_corrections(self, text: str, spelling_issues: str) -> str:
        if "No spelling errors found" in spelling_issues or not spelling_issues.strip():
            return text
        prompt = (
            f"Original text:\n{text}\n\n"
            f"Spelling corrections to apply:\n{spelling_issues}\n\n"
            "Return the text with ONLY these spelling corrections applied. Do not "
            "change grammar, punctuation, wording, or structure. Return only the text."
        )
        return self.spell_correction_agent.inference(prompt).strip()

    # ---- RAG retrieval --------------------------------------------------------
    def _retrieve_rules(self, text: str) -> List[str]:
        if self.retriever is None:
            return []
        try:
            return self.retriever.retrieve(text, k=self.top_k)
        except Exception:
            # Retrieval is an enhancement, not a hard dependency: if the index is
            # missing or errors, fall back to a plain grammar check.
            return []

    # ---- Stage 3: grammar check (RAG-grounded) --------------------------------
    def _run_grammar_check(self, text: str, retrieved_rules: List[str]) -> str:
        rules_block = ""
        if retrieved_rules:
            joined = "\n".join(f"- {r}" for r in retrieved_rules)
            rules_block = f"Relevant style and grammar rules:\n{joined}\n\n"

        prompt = (
            f"{rules_block}"
            "Using the rules above as guidance where relevant, check the grammar of "
            "the text below. Look for subject-verb agreement, verb tense, articles, "
            "confused words, and punctuation.\n\n"
            "For each issue, use exactly this format, separated by a line with '---':\n"
            "Excerpt: <the problematic phrase>\n"
            "Issue: <one sentence>\n"
            "Suggestion: <one specific fix>\n\n"
            "One issue per excerpt. Do not rewrite the whole text. "
            "If there are no issues, reply exactly: No grammar issues found.\n\n"
            f"Text to check:\n{text}"
        )
        return self.grammar_check_agent.inference(prompt).strip()

    def _apply_grammar_corrections(self, text: str, grammar_suggestions: str) -> str:
        if "No grammar issues found" in grammar_suggestions or not grammar_suggestions.strip():
            return text
        prompt = (
            f"Text (already spell-corrected):\n{text}\n\n"
            f"Grammar suggestions:\n{grammar_suggestions}\n\n"
            "Apply these corrections while preserving the author's voice and tone. "
            "Return ONLY the corrected text — no notes, alternatives, or commentary."
        )
        return self.final_correction_agent.inference(prompt).strip()

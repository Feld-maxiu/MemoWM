"""Tests for deterministic pseudo-web evidence checks."""
import unittest

from scripts.verify_qa_against_pseudo_observation_vllm import (
    evidence_is_present,
    normalize_text,
)


class TestPseudoWebVerification(unittest.TestCase):
    def test_normalization_handles_case_unicode_and_spacing(self):
        self.assertEqual(normalize_text("  ＧitHub\nHOME "), "github home")

    def test_exact_contiguous_quote_is_required(self):
        observation = "Title: GitHub\nVisible text:\n- The future of building happens together"
        self.assertTrue(evidence_is_present("The future of building", observation))
        self.assertFalse(evidence_is_present("building future together", observation))

    def test_empty_or_one_character_quote_is_rejected(self):
        self.assertFalse(evidence_is_present("", "Visible text: A"))
        self.assertFalse(evidence_is_present("A", "Visible text: A"))


if __name__ == "__main__":
    unittest.main()

"""
test_engine.py - Unit and Integration Tests for AI Review Insight Analyzer.
Tests multi-provider configuration, prompt loading, schema validation,
JSON fence extraction, CSV parsing, export generation, and error handling.
"""

import json
import os
import unittest
import pandas as pd
from pydantic import ValidationError

import analyzer


class TestAnalyzerEngine(unittest.TestCase):

    def test_01_config_loading(self):
        """Test config.yaml loads properly with required keys."""
        cfg = analyzer.load_config("config.yaml")
        self.assertIn("provider", cfg)
        self.assertIn("model", cfg["provider"])
        self.assertEqual(cfg["provider"]["model"], "models/gemini-3.8-flash")
        self.assertIn("processing", cfg)
        self.assertIn("batch_size", cfg["processing"])
        self.assertEqual(cfg["processing"]["batch_size"], 5)

    def test_02_system_prompt_loading(self):
        """Test system prompt loads from file and contains security guardrails."""
        prompt = analyzer.load_system_prompt("prompts/system_prompt.txt")
        self.assertTrue(len(prompt) > 100)
        self.assertIn("UNTRUSTED INPUT", prompt)
        self.assertIn("SENTIMENT CLASSIFICATION", prompt)
        self.assertIn("JSON", prompt)

    def test_03_text_input_preparation(self):
        """Test parsing newline-separated text into structured reviews."""
        raw_text = "Great sound quality and battery life.\nTerrible customer service and late delivery.\nAverage product."
        reviews, warnings = analyzer.prepare_reviews_from_text(raw_text, max_reviews=10, max_chars=2000)
        self.assertEqual(len(reviews), 3)
        self.assertEqual(reviews[0]["review_id"], "REV-001")
        self.assertEqual(reviews[1]["review_id"], "REV-002")
        self.assertEqual(reviews[2]["review_id"], "REV-003")
        self.assertEqual(len(warnings), 0)

    def test_04_text_input_capping(self):
        """Test that review count is properly capped at max_reviews."""
        raw_text = "\n".join([f"Review number {i}" for i in range(20)])
        reviews, warnings = analyzer.prepare_reviews_from_text(raw_text, max_reviews=5, max_chars=2000)
        self.assertEqual(len(reviews), 5)
        self.assertTrue(any("Capped" in w for w in warnings))

    def test_05_csv_sample_reviews_loading(self):
        """Test sample_reviews.csv has at least 15 valid reviews."""
        df = pd.read_csv("sample_reviews.csv")
        self.assertIn("review", df.columns)
        self.assertGreaterEqual(len(df), 15)
        reviews, warnings = analyzer.prepare_reviews_from_dataframe(df, column_name="review")
        self.assertGreaterEqual(len(reviews), 15)

    def test_06_csv_missing_column_error(self):
        """Test that missing review column raises an informative warning."""
        df = pd.DataFrame({"random_col": ["a", "b"]})
        reviews, warnings = analyzer.prepare_reviews_from_dataframe(df, column_name="review")
        self.assertEqual(len(reviews), 0)
        self.assertTrue(any("not found" in w for w in warnings))

    def test_07_chunking_logic(self):
        """Test that list is chunked according to batch_size."""
        items = [{"review_id": f"REV-{i}"} for i in range(12)]
        chunks = analyzer.chunk_reviews(items, batch_size=5)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 5)
        self.assertEqual(len(chunks[1]), 5)
        self.assertEqual(len(chunks[2]), 2)

    def test_08_extract_json_markdown_fences(self):
        """Test stripping markdown json code fences."""
        raw_with_fences = "```json\n{\"results\": []}\n```"
        parsed = analyzer.extract_json_payload(raw_with_fences)
        self.assertEqual(parsed, {"results": []})

    def test_09_schema_validation_success(self):
        """Test that valid LLM JSON is successfully validated by Pydantic."""
        sample_valid_json = {
            "results": [
                {
                    "review_id": "REV-001",
                    "sentiment": "positive",
                    "explanation": "Customer praises the battery life.",
                    "aspects": ["Battery Life", "Sound Quality"],
                    "positive_points": ["Battery lasts 30 hours", "Crisp sound"],
                    "negative_points": []
                }
            ]
        }
        validated = analyzer.BatchAnalysisResponse.model_validate(sample_valid_json)
        self.assertEqual(len(validated.results), 1)
        self.assertEqual(validated.results[0].sentiment, "positive")

    def test_10_schema_validation_failure(self):
        """Test that malformed JSON missing required fields raises ValidationError."""
        invalid_json = {
            "results": [
                {
                    "review_id": "REV-001",
                    # missing sentiment and explanation
                }
            ]
        }
        with self.assertRaises(ValidationError):
            analyzer.BatchAnalysisResponse.model_validate(invalid_json)

    def test_11_executive_summary_schema(self):
        """Test executive summary validation."""
        valid_summary = {
            "overall_sentiment": "Predominantly positive with minor app stability concerns.",
            "appreciated_features": ["Battery Longevity", "Sound Clarity"],
            "common_complaints": ["App crashes", "Late delivery"],
            "emerging_themes": ["App reliability issues on startup"],
            "actionable_recommendations": [
                "1. Fix companion app crashes",
                "2. Improve shipping partner SLAs",
                "3. Market 30-hour battery capability"
            ],
            "summary_text": "This is a cohesive summary narrative explaining overall metrics."
        }
        validated = analyzer.ExecutiveSummaryModel.model_validate(valid_summary)
        self.assertEqual(len(validated.actionable_recommendations), 3)

    def test_12_metrics_calculation(self):
        """Test metrics calculation counts, percentages, and aspects."""
        mock_results = [
            {
                "review_id": "REV-001",
                "review_text": "Love the battery",
                "sentiment": "positive",
                "explanation": "Praise",
                "aspects": ["Battery Life"],
                "positive_points": ["Long battery"],
                "negative_points": []
            },
            {
                "review_id": "REV-002",
                "review_text": "Terrible screen",
                "sentiment": "negative",
                "explanation": "Complaint",
                "aspects": ["Screen Quality"],
                "positive_points": [],
                "negative_points": ["Screen cracked"]
            },
            {
                "review_id": "REV-003",
                "review_text": "Okay battery, normal screen",
                "sentiment": "neutral",
                "explanation": "Balanced",
                "aspects": ["Battery Life", "Screen Quality"],
                "positive_points": [],
                "negative_points": []
            }
        ]
        metrics = analyzer.calculate_metrics(mock_results)
        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["positive_count"], 1)
        self.assertEqual(metrics["negative_count"], 1)
        self.assertEqual(metrics["neutral_count"], 1)
        self.assertEqual(metrics["aspect_counts"]["Battery Life"], 2)
        self.assertEqual(metrics["aspect_counts"]["Screen Quality"], 2)

    def test_13_exports_generation(self):
        """Test CSV and JSON export byte generation."""
        mock_results = [
            {
                "review_id": "REV-001",
                "review_text": "Great device",
                "sentiment": "positive",
                "explanation": "Praise",
                "aspects": ["Build"],
                "positive_points": ["Solid casing"],
                "negative_points": []
            }
        ]
        csv_bytes = analyzer.results_to_csv_bytes(mock_results)
        json_bytes = analyzer.results_to_json_bytes(mock_results)
        self.assertTrue(len(csv_bytes) > 0)
        self.assertTrue(len(json_bytes) > 0)
        self.assertIn(b"REV-001", csv_bytes)
        self.assertIn(b"Solid casing", json_bytes)

    def test_14_missing_key_behavior_gemini(self):
        """Test get_llm_client raises informative error when Gemini key is missing."""
        old_key = os.environ.get("GEMINI_API_KEY")
        try:
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            cfg = analyzer.load_config("config.yaml")
            with self.assertRaises(ValueError) as ctx:
                analyzer.get_llm_client(provider_name="gemini", api_key=None, config=cfg)
            self.assertIn("Google Gemini API key missing", str(ctx.exception))
            self.assertIn("aistudio.google.com", str(ctx.exception))
        finally:
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_15_missing_key_behavior_openai(self):
        """Test get_llm_client raises informative error when OpenAI key is missing."""
        old_key = os.environ.get("OPENAI_API_KEY")
        try:
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            cfg = analyzer.load_config("config.yaml")
            with self.assertRaises(ValueError) as ctx:
                analyzer.get_llm_client(provider_name="openai", api_key=None, config=cfg)
            self.assertIn("OpenAI API key missing", str(ctx.exception))
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()

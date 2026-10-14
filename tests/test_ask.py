import os
import unittest
from unittest.mock import MagicMock, patch

import httpx
import ollama

import ask


SAMPLE_CHUNK = {
    "title": "เอกสารทดสอบ",
    "content": "เนื้อหาทดสอบ",
    "page_start": 1,
    "page_end": 1,
    "document_year": 2560,
    "tax_year": 2568,
    "similarity": 0.9,
}


class AskTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @patch("ask.ollama.Client")
    def test_local_client_has_finite_default_timeouts(self, client_class):
        ask.make_client()

        timeout = client_class.call_args.kwargs["timeout"]
        self.assertEqual(timeout.connect, 10.0)
        self.assertEqual(timeout.read, 300.0)
        self.assertEqual(timeout.write, 30.0)
        self.assertEqual(timeout.pool, 10.0)

    @patch("ask.ollama.Client")
    def test_hosted_client_uses_api_key_and_custom_timeouts(self, client_class):
        os.environ.update(
            {
                "OLLAMA_API_KEY": "secret",
                "OLLAMA_CONNECT_TIMEOUT": "2.5",
                "OLLAMA_READ_TIMEOUT": "45",
            }
        )

        ask.make_client()

        kwargs = client_class.call_args.kwargs
        self.assertEqual(kwargs["host"], "https://ollama.com")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(kwargs["timeout"].connect, 2.5)
        self.assertEqual(kwargs["timeout"].read, 45.0)

    def test_invalid_timeout_has_clear_configuration_error(self):
        os.environ["OLLAMA_READ_TIMEOUT"] = "never"

        with self.assertRaisesRegex(
            ask.OllamaConfigurationError,
            "OLLAMA_READ_TIMEOUT must be a positive number",
        ):
            ask.get_ollama_timeout()

    def test_hosted_api_requires_model_name(self):
        os.environ["OLLAMA_API_KEY"] = "secret"

        with self.assertRaisesRegex(
            ask.OllamaConfigurationError,
            "OLLAMA_MODEL is required",
        ):
            ask.get_model_name()

    def test_default_temperature_balances_consistency_and_variety(self):
        self.assertEqual(ask.get_temperature(), 0.5)

    def test_temperature_can_be_configured(self):
        os.environ["OLLAMA_TEMPERATURE"] = "1.2"

        self.assertEqual(ask.get_temperature(), 1.2)

    def test_temperature_must_be_in_supported_range(self):
        os.environ["OLLAMA_TEMPERATURE"] = "2.1"

        with self.assertRaisesRegex(
            ask.OllamaConfigurationError,
            "OLLAMA_TEMPERATURE must be between 0 and 2",
        ):
            ask.get_temperature()

    @patch("ask.make_client")
    def test_non_streaming_answer_and_client_cleanup(self, make_client):
        client = make_client.return_value
        client.chat.return_value = {"message": {"content": "คำตอบ [1]"}}

        answer = ask.generate_answer("คำถาม", [SAMPLE_CHUNK], stream=False)

        self.assertEqual(answer, "คำตอบ")
        self.assertEqual(client.chat.call_args.kwargs["options"]["temperature"], 0.5)
        client.close.assert_called_once_with()

    @patch("ask.make_client")
    def test_streaming_answer_is_joined_and_client_is_closed(self, make_client):
        client = make_client.return_value
        client.chat.return_value = iter(
            [
                {"message": {"content": "คำ"}},
                {"message": {"content": "ตอบ"}},
            ]
        )

        with patch("builtins.print"):
            answer = ask.generate_answer("คำถาม", [SAMPLE_CHUNK])

        self.assertEqual(answer, "คำตอบ")
        client.close.assert_called_once_with()

    def test_all_citation_marker_forms_are_removed(self):
        answer = ask.strip_citation_markers("ข้อแรก [1] และข้อสอง [2, 3] [4-5]")

        self.assertEqual(answer, "ข้อแรก และข้อสอง")

    def test_only_two_sources_are_displayed(self):
        chunks = [
            {**SAMPLE_CHUNK, "title": f"เอกสาร {number}"}
            for number in range(1, 4)
        ]

        with patch("builtins.print") as output:
            ask.print_sources(chunks)

        rendered = "\n".join(
            " ".join(str(value) for value in call.args)
            for call in output.call_args_list
        )
        self.assertIn("เอกสาร 1", rendered)
        self.assertIn("เอกสาร 2", rendered)
        self.assertNotIn("เอกสาร 3", rendered)

    @patch("ask.make_client")
    def test_client_is_closed_when_chat_fails(self, make_client):
        client = make_client.return_value
        client.chat.side_effect = httpx.ReadTimeout("slow")

        with self.assertRaises(httpx.ReadTimeout):
            ask.generate_answer("คำถาม", [SAMPLE_CHUNK], stream=False)

        client.close.assert_called_once_with()

    @patch("ask.generate_answer")
    @patch("ask.search_similar_chunks", return_value=[])
    def test_no_chunks_skips_ollama(self, search, generate):
        with patch("builtins.print"):
            result = ask.ask("คำถาม", MagicMock(), MagicMock())

        self.assertEqual(result, {"answer": ask.NOT_FOUND, "sources": []})
        generate.assert_not_called()

    @patch("ask.generate_answer", side_effect=httpx.ReadTimeout("slow"))
    @patch("ask.search_similar_chunks", return_value=[SAMPLE_CHUNK])
    def test_timeout_becomes_actionable_cli_error(self, search, generate):
        with self.assertRaises(SystemExit) as raised:
            ask.ask("คำถาม", MagicMock(), MagicMock())

        self.assertIn("did not respond within 300 seconds", str(raised.exception))
        self.assertIn("OLLAMA_READ_TIMEOUT", str(raised.exception))

    @patch("ask.search_similar_chunks", return_value=[SAMPLE_CHUNK])
    def test_invalid_context_window_becomes_cli_error(self, search):
        os.environ["OLLAMA_NUM_CTX"] = "large"

        with self.assertRaises(SystemExit) as raised:
            ask.ask("คำถาม", MagicMock(), MagicMock())

        self.assertIn("OLLAMA_NUM_CTX must be a positive number", str(raised.exception))

    @patch("ask.generate_answer", side_effect=ConnectionError("offline"))
    @patch("ask.search_similar_chunks", return_value=[SAMPLE_CHUNK])
    def test_local_connection_error_has_startup_hint(self, search, generate):
        with self.assertRaises(SystemExit) as raised:
            ask.ask("คำถาม", MagicMock(), MagicMock())

        self.assertIn("ollama serve", str(raised.exception))

    @patch("ask.generate_answer")
    @patch("ask.search_similar_chunks", return_value=[SAMPLE_CHUNK])
    def test_authentication_error_has_api_key_hint(self, search, generate):
        os.environ["OLLAMA_API_KEY"] = "bad-key"
        generate.side_effect = ollama.ResponseError("unauthorized", status_code=401)

        with self.assertRaises(SystemExit) as raised:
            ask.ask("คำถาม", MagicMock(), MagicMock())

        self.assertIn("rejected the API key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

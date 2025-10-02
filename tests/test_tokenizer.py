import unittest

from vmf_ai.tokenizer import VMFTokenizer


class TestVMFTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = VMFTokenizer()
        self.tokenizer.fit([
            "worldspawn {\n\t""classname"" \"worldspawn\"\n}",
            "entity {\n\t""key"" \"value\"\n}",
        ])

    def test_encode_optional_markers(self) -> None:
        full = self.tokenizer.encode("worldspawn { }")
        without_eos = self.tokenizer.encode("worldspawn { }", add_eos=False)
        self.assertEqual(full[0], self.tokenizer.bos_id)
        self.assertEqual(full[-1], self.tokenizer.eos_id)
        self.assertEqual(without_eos[0], self.tokenizer.bos_id)
        self.assertNotEqual(without_eos[-1], self.tokenizer.eos_id)
        self.assertNotIn(self.tokenizer.eos_id, without_eos)

    def test_decode_stops_at_eos(self) -> None:
        tokens = self.tokenizer.encode("worldspawn { }")
        # Append extra tokens that should never appear in the decoded output.
        extra = self.tokenizer.encode("entity", add_bos=False, add_eos=False)
        decoded = self.tokenizer.decode(tokens + extra)
        self.assertEqual(decoded.strip(), "worldspawn { }")


if __name__ == "__main__":
    unittest.main()

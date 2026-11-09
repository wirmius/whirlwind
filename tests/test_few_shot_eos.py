"""few_shot_eos: the few-shot prompt keeps the simple
style (instruction + 'Q:/A:' blob in the system message, one user question)
— the ONLY delta is the model's DECLARED EOS token appended after each
example, so the in-context pattern becomes 'answer → EOS' (the token the
engine stops on) instead of 'answer → \\n\\nQ:'.
"""

import unittest

TOK = "Qwen/Qwen3-0.6B-Base"


def _make_loader(eos_mode: bool):
    """DataLoaderNode shell with a tiny fake dataset (no cluster, no I/O)."""
    from whirlwind.config import DataLoaderConfig
    from whirlwind.data.dataloader_node import DataLoaderNode

    node = object.__new__(DataLoaderNode)
    node.config = DataLoaderConfig(
        default_dataset="train",
        system_prompt_template=(
            'Solve the following math problems. Always end your answer with '
            '"#### {{answer}}" where {{answer}} is the final numerical result.\n\n'
            "{few_shot_examples}"
        ),
        few_shot_count=2,
        few_shot_eos=eos_mode,
    )
    rows = [
        {"prompt": f"Question number {i}?", "response": f"Reasoning {i}. #### {i * 11}"}
        for i in range(8)
    ]

    class _Handle:
        config = {"seed": 42, "prompt_key": "prompt", "response_key": "response"}

        class _Ds:
            def __init__(self, rows):
                self.rows = rows
                self.column_names = ["prompt", "response"]

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, idx):
                return self.rows[idx]

        _dataset = None

    h = _Handle()
    h._dataset = _Handle._Ds(rows)
    node._datasets = {"train": h}
    return node


class TestFewShotEosBuild(unittest.TestCase):
    def test_legacy_blob_bit_identical(self):
        node = _make_loader(eos_mode=False)
        sys_prompt = node._build_system_prompt()
        self.assertIsNone(node._prefix_messages)
        # blob form: Q:/A: prefixed examples inside the system prompt
        self.assertIn("Q: Question number", sys_prompt)
        self.assertIn("A: Reasoning", sys_prompt)
        self.assertIn("#### ", sys_prompt)
        self.assertEqual(sys_prompt.count("\n\nQ: "), 2)

    def test_eos_mode_transports_raw_turns(self):
        node = _make_loader(eos_mode=True)
        sys_prompt = node._build_system_prompt()
        # system = the instruction ONLY (examples re-attached rollout-side)
        self.assertIn("Solve the following math problems", sys_prompt)
        self.assertNotIn("Q: Question number", sys_prompt)
        # transport: alternating raw user/assistant contents, deterministic
        pm = node._prefix_messages
        self.assertIsNotNone(pm)
        self.assertEqual(len(pm), 4)
        self.assertEqual([m["role"] for m in pm], ["user", "assistant"] * 2)
        self.assertTrue(all(m["content"].startswith("Question number ") for m in pm[::2]))
        self.assertTrue(all("#### " in m["content"] for m in pm[1::2]))
        # same seed → same examples as the blob mode
        legacy = _make_loader(eos_mode=False)._build_system_prompt()
        for m in pm[::2]:
            self.assertIn(m["content"], legacy)


class TestFewShotEosTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(TOK)

    def _rollout(self, use_template=True):
        from whirlwind.rollout.rollout_sglang import RolloutNode

        r = object.__new__(RolloutNode)
        r._tokenizer = self.tok
        r._use_chat_template = use_template
        return r

    PREFIX = [
        {"role": "user", "content": "What is 6 * 7?"},
        {"role": "assistant", "content": "Six times seven is 42. #### 42"},
    ]

    def test_old_style_blob_with_declared_eos(self):
        r = self._rollout()
        out = r._apply_chat_template(
            "What is 2 + 2?", "Solve. End with #### n.", self.PREFIX
        )
        eos = self.tok.eos_token
        # OLD structure: ONE system message (instruction + Q:/A: blob with the
        # declared EOS after the example) + ONE user message + assistant header
        self.assertEqual(out.count("<|im_start|>user"), 1)
        self.assertEqual(out.count("<|im_start|>system"), 1)
        self.assertIn(
            f"Solve. End with #### n.\n\n"
            f"Q: What is 6 * 7?\nA: Six times seven is 42. #### 42{eos}",
            out,
        )
        self.assertTrue(
            out.endswith("<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n")
        )
        # the EOS string lands as the REAL declared eos token id
        ids = self.tok.encode(out, add_special_tokens=False)
        self.assertIn(self.tok.eos_token_id, ids)

    def test_legacy_path_unchanged(self):
        r = self._rollout()
        out = r._apply_chat_template("Q?", "sys")
        self.assertEqual(
            out,
            self.tok.apply_chat_template(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Q?"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            ),
        )

    def test_flat_fallback_keeps_eos_demonstration(self):
        r = self._rollout(use_template=False)
        out = r._apply_chat_template("Q?", "sys", self.PREFIX)
        eos = self.tok.eos_token
        self.assertEqual(
            out,
            f"sys\n\nQ: What is 6 * 7?\nA: Six times seven is 42. #### 42{eos}\n\nQ?",
        )


if __name__ == "__main__":
    unittest.main()


class TestPrefixMessagesTransport(unittest.TestCase):
    """Regression: prefix_messages must survive
    the dataloader → BatchResult → to_prompt_batch hop. It was silently
    dropped here (field not on BatchResult) and the model trained on an
    instruction-only prompt: eot 61%→7%, 7000-token example-free rambles."""

    def test_batch_result_roundtrip_keeps_prefix_messages(self):
        from whirlwind.messages import BatchResult, PromptBatch, _to_dict, _from_dict

        pm = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1 #### 11"},
        ]
        # the dataloader handler returns the PromptBatch; the orchestrator
        # reconstructs a BatchResult from the RPC dict (field-filtered!)
        pb = PromptBatch(
            prompts=["q"], responses=["11"], system_prompt="instruction",
            prefix_messages=pm, step=3, entry_ids=[0],
        )
        raw = _to_dict(pb)
        br = _from_dict(BatchResult, raw)
        self.assertIsNotNone(br.prefix_messages, "BatchResult dropped prefix_messages")
        out = br.to_prompt_batch()
        self.assertEqual(out.prefix_messages, pm)
        self.assertEqual(out.system_prompt, "instruction")
        # and the full orchestrator→rollout hop
        raw2 = _to_dict(out)
        out2 = _from_dict(PromptBatch, raw2)
        self.assertEqual(out2.prefix_messages, pm)


import sys
import unittest
from pathlib import Path

class TestKernelArgStripping(unittest.TestCase):
    def test_strip_colab_kernel_args(self):
        from benchmark_agentic_llm import strip_kernel_args
        
        # Scenario 1: Standalone -f followed by json file
        raw_1 = ["-f", "/root/.local/share/jupyter/runtime/kernel-17af4e42-f4f4-42d8-b8f0-44dbee6e130f.json"]
        self.assertEqual(strip_kernel_args(raw_1), [])

        # Scenario 2: Direct json file passed without -f (the exact error from user)
        raw_2 = ["/root/.local/share/jupyter/runtime/kernel-17af4e42-f4f4-42d8-b8f0-44dbee6e130f.json"]
        self.assertEqual(strip_kernel_args(raw_2), [])

        # Scenario 3: ipykernel flags plus kernel file
        raw_3 = [
            "--ip=127.0.0.1",
            "--stdin=9000",
            "--control=9001",
            "--hb=9002",
            "--Session.signature_scheme=\"hmac-sha256\"",
            "/root/.local/share/jupyter/runtime/kernel-test.json",
            "sweep",
            "--preset", "quick"
        ]
        self.assertEqual(strip_kernel_args(raw_3), ["sweep", "--preset", "quick"])

    def test_parser_notebook_fallback(self):
        from benchmark_agentic_llm import parse_args
        
        # Test calling parse_args with typical Colab kernel argv
        simulated_colab_argv = ["/root/.local/share/jupyter/runtime/kernel-17af4e42-f4f4-42d8-b8f0-44dbee6e130f.json"]
        # Should default to sweep and not crash!
        args = parse_args(simulated_colab_argv)
        self.assertEqual(args.command, "sweep")
        self.assertEqual(args.preset, "quick")

    def test_sweet_spot_synthesis(self):
        from benchmark_agentic_llm import rank_summaries, synthesize_sweet_spot
        
        sample_rows = [
            {
                "model": "ornith",
                "profile": "speed32_q4",
                "status": "ok",
                "agent_pass_rate": 0.75,
                "median_decode_tps": 22.5,
                "median_prompt_tps": 120.0,
                "median_ttft_s": 0.45,
                "min_vram_headroom_mb": 4000.0,
                "ctx": 32768,
                "batch": 2048,
                "ubatch": 512,
                "cache_k": "q4_0",
                "cache_v": "q4_0",
                "split_mode": "layer",
            },
            {
                "model": "qwen",
                "profile": "balanced64",
                "status": "ok",
                "agent_pass_rate": 1.0,
                "median_decode_tps": 24.0,
                "median_prompt_tps": 135.0,
                "median_ttft_s": 0.38,
                "min_vram_headroom_mb": 2500.0,
                "ctx": 65536,
                "batch": 2048,
                "ubatch": 512,
                "cache_k": "q4_0",
                "cache_v": "q4_0",
                "split_mode": "layer",
            },
        ]
        ranked = rank_summaries(sample_rows)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["model"], "qwen")
        
        synthesis = synthesize_sweet_spot(ranked)
        self.assertIn("ornith", synthesis["per_model_sweet_spot"])
        self.assertIn("qwen", synthesis["per_model_sweet_spot"])
        self.assertIn("ornith", synthesis["suggested_launcher_configs"])
        self.assertIn("qwen", synthesis["suggested_launcher_configs"])
        self.assertEqual(synthesis["suggested_launcher_configs"]["qwen"]["context_size"], 65536)

    def test_find_model_explicit_or_fallback(self):
        from benchmark_agentic_llm import find_kaggle_model
        # Passing non-existent explicit path returns None
        self.assertIsNone(find_kaggle_model("ornith", "/non/existent/path.gguf"))

    def test_clean_args_dict_and_json_serialization(self):
        import tempfile
        from benchmark_agentic_llm import clean_args_dict, save_results, parse_args

        def dummy_func():
            pass

        # Simulate args with a function and a Path object
        args = parse_args(["sweep", "--preset", "quick"])
        args.custom_func = dummy_func
        args.custom_path = Path("/dummy/path")

        cleaned = clean_args_dict(args)
        self.assertEqual(cleaned["custom_func"], "dummy_func")
        self.assertEqual(cleaned["custom_path"], "/dummy/path")

        # Verify save_results can serialize raw with function without error
        with tempfile.TemporaryDirectory() as td:
            raw = {"args": cleaned, "runs": []}
            save_results(Path(td), raw, [], quiet=True)
            results_path = Path(td) / "results.json"
            self.assertTrue(results_path.exists())
            self.assertIn('"dummy_func"', results_path.read_text(encoding="utf-8"))

    def test_build_server_env(self):
        from benchmark_agentic_llm import build_server_env
        env = build_server_env(workdir=Path("/dummy"), server_bin=Path("/dummy/bin/llama-server"))
        self.assertIn("LD_LIBRARY_PATH", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["GGML_CUDA_NO_VMM"], "1")


if __name__ == "__main__":
    unittest.main()

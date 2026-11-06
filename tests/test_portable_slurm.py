"""Portable wrapper checks using synthetic files and a recorder, never training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.slurm"


class PortableSlurmTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project with spaces"
        for name in ("pocketgpt", "scripts", "tests"):
            (self.project / name).mkdir(parents=True)
        for name in ("pocketgpt/__main__.py", "scripts/server_preflight.py"):
            (self.project / name).write_text("# Synthetic wrapper fixture only\n")
        self.recipe = self.project / "recipe.json"
        self.recipe.write_text("{}\n")
        self.calls = self.root / "calls.jsonl"
        self.interpreter = self.root / "recording interpreter"
        self.interpreter.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, sys\n"
            "names = ['WANDB_MODE', 'WANDB_DISABLED', 'DO_NOT_TRACK', 'HF_HUB_OFFLINE', "
            "'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONPATH', 'PYTHONHOME', 'OMP_NUM_THREADS']\n"
            "with open(os.environ['POCKETGPT_WRAPPER_CALLS'], 'a') as out:\n"
            "    out.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
            "'env': {name: os.environ.get(name) for name in names}}) + '\\n')\n"
        )
        self.interpreter.chmod(0o700)
        self.environment = dict(os.environ, SLURM_JOB_ID="synthetic-wrapper-test",
                                SLURM_CPUS_PER_TASK="4", POCKETGPT_PROJECT=str(self.project),
                                POCKETGPT_PYTHON=str(self.interpreter),
                                POCKETGPT_WRAPPER_CALLS=str(self.calls))

    def run_wrapper(self, *args, env=None):
        return subprocess.run(["/bin/bash", str(SCRIPT), *map(str, args)],
                              env=env or self.environment, capture_output=True, text=True,
                              timeout=20)

    def test_requires_allocation(self):
        environment = dict(self.environment)
        environment.pop("SLURM_JOB_ID", None)
        result = self.run_wrapper(self.recipe, env=environment)
        self.assertEqual(result.returncode, 2)
        self.assertIn("login node", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_requires_absolute_recipe_and_explicit_environment(self):
        self.assertEqual(self.run_wrapper().returncode, 2)
        self.assertEqual(self.run_wrapper("relative.json").returncode, 2)
        environment = dict(self.environment)
        environment.pop("POCKETGPT_PYTHON")
        self.assertEqual(self.run_wrapper(self.recipe, env=environment).returncode, 2)
        self.assertFalse(self.calls.exists())

    def test_rejects_bad_or_unknown_options_without_executing(self):
        for options in (("--resume", "relative.pt"), ("--max-steps", "0"),
                        ("--unknown",), ("--resume",), ("--max-steps",)):
            with self.subTest(options=options):
                self.assertEqual(self.run_wrapper(self.recipe, *options).returncode, 2)
        self.assertFalse(self.calls.exists())

    def test_empty_optional_arrays_and_environment_sanitization(self):
        environment = dict(self.environment, PYTHONPATH="must-be-cleared",
                           PYTHONHOME="must-be-cleared")
        result = self.run_wrapper(self.recipe, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["args"][0], "scripts/server_preflight.py")
        self.assertEqual(calls[1]["args"], ["-m", "unittest", "discover", "-s", "tests", "-v"])
        self.assertEqual(calls[2]["args"], ["-m", "pocketgpt", "train", "--config",
                                             str(self.recipe), "--device", "cuda"])
        for call in calls:
            self.assertEqual(call["cwd"], str(self.project))
            self.assertIsNone(call["env"]["PYTHONPATH"])
            self.assertIsNone(call["env"]["PYTHONHOME"])
            self.assertEqual(call["env"]["WANDB_MODE"], "disabled")
            self.assertEqual(call["env"]["WANDB_DISABLED"], "true")
            self.assertEqual(call["env"]["DO_NOT_TRACK"], "1")
            self.assertEqual(call["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(call["env"]["OMP_NUM_THREADS"], "4")

    def test_resume_and_step_flags_are_forwarded_without_losing_spaces(self):
        checkpoint = self.project / "run with spaces" / "last.pt"
        result = self.run_wrapper(self.recipe, "--resume", checkpoint, "--max-steps", "10")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line)["args"] for line in self.calls.read_text().splitlines()]
        self.assertEqual(calls[0][-2:], ["--resume", str(checkpoint)])
        self.assertEqual(calls[2][-4:], ["--resume", str(checkpoint), "--max-steps", "10"])
        self.assertNotIn("--max-steps", calls[0])


if __name__ == "__main__":
    unittest.main()

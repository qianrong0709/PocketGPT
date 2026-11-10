import errno
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch

from pocketgpt.checkpoint import rng_state, save_checkpoint, seed_all


class CheckpointIOTests(unittest.TestCase):
    def payload(self, step):
        return {"global_step": step, "model": {"weight": torch.arange(12).reshape(3, 4)},
                "optimizer": {"state": {}}, "rng": rng_state()}

    def test_stream_roundtrip_previous_and_legacy_temporary_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            legacy_temporary = path.with_suffix(".pt.tmp")
            legacy_temporary.write_bytes(b"retained recovery evidence")
            save_checkpoint(path, self.payload(10))
            save_checkpoint(path, self.payload(20))
            self.assertEqual(torch.load(path, weights_only=True)["global_step"], 20)
            self.assertEqual(torch.load(path.with_name("last.pt.previous"), weights_only=True)["global_step"], 10)
            self.assertEqual(legacy_temporary.read_bytes(), b"retained recovery evidence")

    def test_python_stream_and_no_training_rng_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_all(23)
            payload = self.payload(1)
            original_save = torch.save
            streams = []
            def serialize(value, handle):
                self.assertTrue(hasattr(handle, "write"))
                streams.append(handle)
                return original_save(value, handle)
            with patch("pocketgpt.checkpoint.torch.save", side_effect=serialize):
                save_checkpoint(Path(directory) / "last.pt", payload)
            current = rng_state()
            for key in ("python", "numpy"):
                self.assertEqual(current[key], payload["rng"][key])
            self.assertTrue(torch.equal(current["torch"], payload["rng"]["torch"]))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(current["cuda"], payload["rng"]["cuda"])))
            self.assertTrue(all(handle.closed for handle in streams))

    def test_single_retry_recovers_transient_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            original_save = torch.save
            attempts = []
            def serialize(value, handle):
                attempts.append(handle.name)
                if len(attempts) == 1:
                    raise OSError(errno.EIO, "synthetic transient failure")
                return original_save(value, handle)
            with patch("pocketgpt.checkpoint.torch.save", side_effect=serialize), self.assertWarns(RuntimeWarning):
                save_checkpoint(path, self.payload(2))
            self.assertEqual(len(attempts), 2)
            self.assertEqual(torch.load(path, weights_only=True)["global_step"], 2)
            self.assertEqual(len(list(path.parent.glob(".last.pt.*.tmp"))), 1)

    def test_persistent_failure_preserves_last_and_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, self.payload(1))
            save_checkpoint(path, self.payload(2))
            before = path.read_bytes()
            previous = path.with_name("last.pt.previous").read_bytes()
            with patch("pocketgpt.checkpoint.torch.save", side_effect=OSError(errno.ENOSPC, "synthetic full device")) as serialize:
                with warnings.catch_warnings(), self.assertRaisesRegex(RuntimeError, "errno=28"):
                    warnings.simplefilter("ignore")
                    save_checkpoint(path, self.payload(3))
            self.assertEqual(serialize.call_count, 2)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.with_name("last.pt.previous").read_bytes(), previous)

    def test_failed_validation_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, self.payload(1))
            before = path.read_bytes()
            with patch("pocketgpt.checkpoint.torch.load", return_value={"global_step": -1}):
                with warnings.catch_warnings(), self.assertRaisesRegex(RuntimeError, "phase=verify"):
                    warnings.simplefilter("ignore")
                    save_checkpoint(path, self.payload(2))
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

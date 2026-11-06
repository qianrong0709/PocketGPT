"""Runs ONLY inside the user's original DeeplyTough environment (Python >=3.6).

Uses the original preprocessing, checkpoint, and L2-normalization. No downloads.
API reference: BenevolentAI/DeeplyTough engine/predictor.py and misc/utils.py.
This bridge is not a replacement encoder and never falls back to random weights.
"""
import argparse
import os
import shutil
import sys
import shlex
import subprocess


class ExplicitMGLSubprocess:
    """Narrow adapter for the original helper command, not its chemistry.

    The original utils module calls bash -ic/source activate with a hard-coded
    environment name. Translate ONLY that exact command to the user-selected
    original MGLTools helper. All other subprocess calls retain their behavior.
    """
    def __init__(self, delegate, prefix):
        self.delegate = delegate
        self.prefix = os.path.realpath(prefix)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def run(self, command, *args, **kwargs):
        expected = ['source', 'activate', 'deeplytough_mgltools;', 'pythonsh',
                    '$CONDA_PREFIX/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py']
        if (isinstance(command, (list, tuple)) and len(command) == 3 and
                list(command[:2]) == ['/bin/bash', '-ic']):
            try:
                tokens = shlex.split(command[2])
            except ValueError:
                tokens = []
            if (tokens[:5] == expected and len(tokens) == 11 and tokens[5] == '-r' and
                    tokens[7:] == ['-U', 'nphs_lps_waters', '-A', 'hydrogens']):
                helper = os.path.join(self.prefix, 'bin', 'pythonsh')
                script = os.path.join(self.prefix, 'MGLToolsPckgs', 'AutoDockTools',
                                      'Utilities24', 'prepare_receptor4.py')
                environment = dict(kwargs.get('env') or os.environ)
                environment.update(CONDA_PREFIX=self.prefix, MGLHOME=self.prefix,
                                   PATH=os.path.join(self.prefix, 'bin') + os.pathsep + environment.get('PATH', ''))
                kwargs['env'] = environment
                return self.delegate.run([helper, script] + tokens[5:], *args, **kwargs)
        return self.delegate.run(command, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "weights", "protein", "pocket", "work", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mgltools-prefix", required=True)
    args = parser.parse_args()
    # The original tree mixes `engine`/`misc` imports with
    # `deeplytough.misc` imports. Both explicit roots are needed under -I.
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, "deeplytough"))
    import numpy as np
    import torch
    from engine.datasets import PointOfInterestVoxelizedDataset
    from engine.models import create_model
    from engine.predictor import load_and_precompute_point_feats
    from misc import utils as original_utils
    from misc.utils import center_from_pdb_file, htmd_featurizer
    original_utils.subprocess = ExplicitMGLSubprocess(subprocess, args.mgltools_prefix)

    torch.set_num_threads(1)
    # Only local, user-selected encoder checkpoints are accepted. Old releases
    # store an argparse.Namespace; newer torch needs an explicit safe global.
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([argparse.Namespace]):
            state = torch.load(args.weights, map_location="cpu", weights_only=True)
    else:
        state = torch.load(args.weights, map_location="cpu")
    model = create_model(state["args"], PointOfInterestVoxelizedDataset, torch.device("cpu"))
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    # Original preprocessing embeds paths in a shell command. Use fixed safe
    # filenames within a private work directory, not arbitrary user filenames.
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-." for c in args.work):
        raise ValueError("DeeplyTough work path must not contain shell metacharacters or spaces")
    protein = os.path.join(args.work, "protein.pdb")
    shutil.copyfile(args.protein, protein)
    entry = {"protein": protein, "pocket": args.pocket,
             "protein_htmd": os.path.join(args.work, "protein.npz"), "key": "input"}
    htmd_featurizer([entry], skip_existing=True)
    if not os.path.isfile(entry["protein_htmd"]):
        raise RuntimeError("Original HTMD/MGLTools preprocessing did not produce protein.npz")
    center = center_from_pdb_file(args.pocket)
    if center is None:
        raise ValueError("Cannot determine the pocket center from this PDB")
    feats = load_and_precompute_point_feats(model, state["args"], [entry], [[center]], "cpu", 0, 1)
    if len(feats) != 1 or feats[0] is None:
        raise RuntimeError("Encoder returned no descriptor")
    vector = np.asarray(feats[0]).reshape(-1).astype(np.float32)
    if vector.shape != (128,) or not np.isfinite(vector).all():
        raise ValueError("Encoder checkpoint did not produce a finite 128-dimensional descriptor")
    temporary = args.output + ".tmp"
    with open(temporary, "wb") as handle:
        np.save(handle, vector, allow_pickle=False)
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()

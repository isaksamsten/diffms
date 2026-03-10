"""Compute evaluation metrics from partial DiffMS test predictions.

When a test run is interrupted or still in progress, this script computes
the same metrics that on_test_epoch_end would report, using whatever
pred/true pickle files have been written so far.

Metrics computed:
  - Top-k accuracy (exact InChI match) for k=1..num_samples
  - Top-k Tanimoto similarity for k=1..num_samples
  - Top-k Cosine similarity for k=1..num_samples
  - Validity (fraction of generated molecules that are valid)

Usage:
    python scripts/eval_partial_preds.py <preds_dir> [--prefix PREFIX] [--max_k 10]
    python scripts/eval_partial_preds.py preds/ --prefix "dfm+cross_entropy_resume_rank_0"
    python scripts/eval_partial_preds.py preds/ --max_k 5 --csv results.csv
"""

import argparse
import glob
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import List

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")




def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def is_valid(mol):
    smiles = mol2smiles(mol)
    if smiles is None:
        return False
    try:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return False
    return len(frags) <= 1


try:
    from rdkit.Chem.MolStandardize.tautomer import (
        TautomerCanonicalizer,
        TautomerTransform,
    )

    _RD_TAUTOMER_CANONICALIZER = "v1"
    _TAUTOMER_TRANSFORMS = (
        TautomerTransform(
            "1,3 heteroatom H shift",
            "[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]",
        ),
        TautomerTransform("1,3 (thio)keto/enol r", "[O,S,Se,Te;X2!H0]-[C]=[C]"),
    )
except ModuleNotFoundError:
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator

    _RD_TAUTOMER_CANONICALIZER = "v2"


def canonical_mol_from_inchi(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    if _RD_TAUTOMER_CANONICALIZER == "v1":
        _molvs_t = TautomerCanonicalizer(transforms=_TAUTOMER_TRANSFORMS)
        mol = _molvs_t.canonicalize(mol)
    else:
        _te = TautomerEnumerator()
        mol = _te.Canonicalize(mol)
    return mol




class AccuracyAccumulator:
    """Top-k exact InChI match accuracy."""

    def __init__(self, max_k: int):
        self.max_k = max_k
        self.correct = [0] * max_k  # correct[i] = matches within top-(i+1)
        self.total = 0

    def update(self, generated_mols: List, true_mol):
        inchis = [Chem.MolToInchi(m) for m in generated_mols if is_valid(m)]
        # deduplicate by frequency (most common first)
        inchi_counter = Counter(inchis)
        unique_inchis = [item for item, _ in inchi_counter.most_common()]

        true_inchi = Chem.MolToInchi(true_mol)
        self.total += 1

        for k in range(self.max_k):
            if true_inchi in unique_inchis[: k + 1]:
                self.correct[k] += 1

    def compute(self):
        if self.total == 0:
            return {}
        return {
            f"acc_at_{k+1}": self.correct[k] / self.total
            for k in range(self.max_k)
        }


class SimilarityAccumulator:
    """Top-k max Tanimoto and Cosine similarity."""

    def __init__(self, max_k: int):
        self.max_k = max_k
        self.tanimoto_sums = [0.0] * max_k
        self.cosine_sums = [0.0] * max_k
        self.total = 0

    def update(self, generated_mols: List, true_mol):
        inchis = [Chem.MolToInchi(m) for m in generated_mols if is_valid(m)]
        inchi_counter = Counter(inchis)
        unique_inchis = [item for item, _ in inchi_counter.most_common()]
        processed_mols = [canonical_mol_from_inchi(inchi) for inchi in unique_inchis]

        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
        self.total += 1

        tani_sims = []
        cos_sims = []
        for mol in processed_mols:
            if mol is None:
                tani_sims.append(0.0)
                cos_sims.append(0.0)
                continue
            try:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                tani_sims.append(DataStructs.TanimotoSimilarity(gen_fp, true_fp))
                cos_sims.append(DataStructs.CosineSimilarity(gen_fp, true_fp))
            except Exception:
                tani_sims.append(0.0)
                cos_sims.append(0.0)

        running_max_tani = 0.0
        running_max_cos = 0.0
        for k in range(self.max_k):
            if k < len(tani_sims):
                running_max_tani = max(running_max_tani, tani_sims[k])
                running_max_cos = max(running_max_cos, cos_sims[k])
            self.tanimoto_sums[k] += running_max_tani
            self.cosine_sums[k] += running_max_cos

    def compute(self):
        if self.total == 0:
            return {}
        result = {}
        for k in range(self.max_k):
            result[f"tanimoto_at_{k+1}"] = self.tanimoto_sums[k] / self.total
            result[f"cosine_at_{k+1}"] = self.cosine_sums[k] / self.total
        return result


class ValidityAccumulator:
    """Fraction of generated molecules that are valid."""

    def __init__(self):
        self.valid = 0
        self.total = 0

    def update(self, generated_mols: List):
        for mol in generated_mols:
            if is_valid(mol):
                self.valid += 1
            self.total += 1

    def compute(self):
        if self.total == 0:
            return 0.0
        return self.valid / self.total




def discover_batches(preds_dir: str, prefix: str = None):
    """Find all (pred, true) pickle pairs in a preds directory.

    Returns list of (batch_idx, pred_path, true_path) sorted by batch_idx.
    """
    preds_dir = Path(preds_dir)
    if prefix:
        pred_pattern = str(preds_dir / f"{prefix}_pred_*.pkl")
    else:
        pred_pattern = str(preds_dir / "*_pred_*.pkl")

    pred_files = sorted(glob.glob(pred_pattern))
    batches = []

    for pred_path in pred_files:
        match = re.search(r"_pred_(\d+)\.pkl$", pred_path)
        if match is None:
            continue
        batch_idx = int(match.group(1))
        true_path = pred_path.replace(f"_pred_{batch_idx}.pkl", f"_true_{batch_idx}.pkl")
        if Path(true_path).exists():
            batches.append((batch_idx, pred_path, true_path))

    batches.sort(key=lambda x: x[0])
    return batches


def load_batch(pred_path: str, true_path: str):
    with open(pred_path, "rb") as f:
        preds = pickle.load(f)
    with open(true_path, "rb") as f:
        trues = pickle.load(f)
    return preds, trues




def main():
    parser = argparse.ArgumentParser(
        description="Compute eval metrics from partial DiffMS test predictions"
    )
    parser.add_argument("preds_dir", type=str, help="Path to predictions directory")
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="File prefix (e.g. 'dfm+cross_entropy_resume_rank_0'). "
        "If not set, all *_pred_*.pkl files are used.",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=None,
        help="Maximum k for top-k metrics (default: inferred from num samples)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optionally write results to a CSV file",
    )
    args = parser.parse_args()

    batches = discover_batches(args.preds_dir, args.prefix)
    if not batches:
        print(f"No prediction files found in {args.preds_dir}")
        if args.prefix:
            print(f"  (looked for prefix '{args.prefix}')")
        return

    print(f"Found {len(batches)} batch(es): indices {[b[0] for b in batches]}")

    first_preds, _ = load_batch(batches[0][1], batches[0][2])
    num_samples = len(first_preds[0]) if first_preds and first_preds[0] else 1
    max_k = args.max_k if args.max_k is not None else num_samples
    max_k = min(max_k, num_samples)

    print(f"Samples per molecule: {num_samples}, computing metrics up to k={max_k}")
    print()

    acc = AccuracyAccumulator(max_k)
    sim = SimilarityAccumulator(max_k)
    val = ValidityAccumulator()

    total_samples = 0
    for batch_idx, pred_path, true_path in batches:
        preds, trues = load_batch(pred_path, true_path)
        n = len(trues)
        total_samples += n
        print(f"  Batch {batch_idx}: {n} samples", end="", flush=True)

        for idx in range(n):
            true_mol = trues[idx]
            pred_mols = preds[idx]

            if true_mol is None:
                print(f"\n    WARNING: true_mol is None at batch {batch_idx}, sample {idx}, skipping")
                continue

            acc.update(pred_mols, true_mol)
            sim.update(pred_mols, true_mol)
            val.update(pred_mols)

        print(" ✓")

    print(f"\nTotal test samples evaluated: {total_samples}")
    print(f"{'=' * 60}")

    acc_results = acc.compute()
    sim_results = sim.compute()
    validity = val.compute()

    display_ks = sorted(set([1, 5, 10, 20, 50, 100, max_k]) & set(range(1, max_k + 1)))

    print(f"\n{'Metric':<25} {'Value':>10}")
    print(f"{'-' * 25} {'-' * 10}")
    print(f"{'validity':<25} {validity:>10.4f}")

    print()
    for k in display_ks:
        key = f"acc_at_{k}"
        if key in acc_results:
            print(f"{'acc@' + str(k):<25} {acc_results[key]:>10.4f}")

    print()
    for k in display_ks:
        key = f"tanimoto_at_{k}"
        if key in sim_results:
            print(f"{'tanimoto@' + str(k):<25} {sim_results[key]:>10.4f}")

    print()
    for k in display_ks:
        key = f"cosine_at_{k}"
        if key in sim_results:
            print(f"{'cosine@' + str(k):<25} {sim_results[key]:>10.4f}")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_samples", total_samples])
            writer.writerow(["num_batches", len(batches)])
            writer.writerow(["validity", f"{validity:.6f}"])
            for k in range(1, max_k + 1):
                writer.writerow([f"acc_at_{k}", f"{acc_results[f'acc_at_{k}']:.6f}"])
            for k in range(1, max_k + 1):
                writer.writerow(
                    [f"tanimoto_at_{k}", f"{sim_results[f'tanimoto_at_{k}']:.6f}"]
                )
            for k in range(1, max_k + 1):
                writer.writerow(
                    [f"cosine_at_{k}", f"{sim_results[f'cosine_at_{k}']:.6f}"]
                )
        print(f"\nResults saved to {args.csv}")


if __name__ == "__main__":
    main()


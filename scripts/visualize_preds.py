"""Visualize predicted vs true molecules from DiffMS test predictions.

Usage:
    python scripts/visualize_preds.py <preds_dir> [--batch 0] [--sample 0] [--top_k 5] [--out output.png]
"""

import argparse
import pickle
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
from PIL import Image, ImageDraw, ImageFont


def safe_mol(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def tanimoto_similarity(mol_a, mol_b):
    mol_a, mol_b = safe_mol(mol_a), safe_mol(mol_b)
    if mol_a is None or mol_b is None:
        return 0.0
    try:
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return 0.0


def mol_to_smiles(mol):
    mol = safe_mol(mol)
    if mol is None:
        return "Invalid"
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return "Invalid"


def load_batch(preds_dir, batch_idx, prefix="dfm+cross_entropy_resume_rank_0"):
    preds_dir = Path(preds_dir)
    pred_file = preds_dir / f"{prefix}_pred_{batch_idx}.pkl"
    true_file = preds_dir / f"{prefix}_true_{batch_idx}.pkl"

    with open(pred_file, "rb") as f:
        preds = pickle.load(f)
    with open(true_file, "rb") as f:
        trues = pickle.load(f)

    return preds, trues


def visualize_sample(true_mol, pred_mols, top_k=5, img_size=(350, 350)):
    top_k = min(top_k, len(pred_mols))

    scored = []
    for mol in pred_mols:
        s_mol = safe_mol(mol)
        if s_mol is not None:
            sim = tanimoto_similarity(true_mol, s_mol)
            scored.append((s_mol, sim))
        else:
            scored.append((None, 0.0))

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:top_k]

    legends = [f"TRUE\n{mol_to_smiles(true_mol)}"]
    mols = [true_mol]

    for mol, sim in scored:
        smiles = mol_to_smiles(mol)
        legend = f"Pred (sim={sim:.3f})\n{smiles}"
        legends.append(legend)
        mols.append(mol)

    n_cols = len(mols)
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=img_size,
        legends=[l.split("\n")[0] for l in legends],
    )

    return img, legends


def main():
    parser = argparse.ArgumentParser(description="Visualize DiffMS predictions")
    parser.add_argument("preds_dir", type=str, help="Path to predictions directory")
    parser.add_argument("--batch", type=int, default=0, help="Batch index to visualize")
    parser.add_argument("--sample", type=int, default=None, help="Sample index within batch (default: show first 4)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to show")
    parser.add_argument("--out", type=str, default="preds_vis.png", help="Output image path")
    parser.add_argument("--prefix", type=str, default="dfm+cross_entropy_resume_rank_0",
                        help="File prefix for pred/true pkl files")
    args = parser.parse_args()

    preds, trues = load_batch(args.preds_dir, args.batch, args.prefix)
    print(len(trues))

    if args.sample is not None:
        samples = [args.sample]
    else:
        samples = list(range(len(trues)))

    all_images = []
    for s_idx in samples:
        true_mol = trues[s_idx]
        pred_mols = preds[s_idx]
        img, legends = visualize_sample(true_mol, pred_mols, top_k=args.top_k)
        all_images.append((img, legends, s_idx))

    if len(all_images) == 1:
        final_img = all_images[0][0]
    else:
        widths = [img.size[0] for img, _, _ in all_images]
        heights = [img.size[1] for img, _, _ in all_images]
        label_height = 30
        total_height = sum(heights) + label_height * len(all_images)
        max_width = max(widths)

        final_img = Image.new("RGB", (max_width, total_height), "white")
        draw = ImageDraw.Draw(final_img)
        y_offset = 0
        for img, legends, s_idx in all_images:
            draw.text((10, y_offset), f"Sample {s_idx}", fill="black")
            y_offset += label_height
            final_img.paste(img, (0, y_offset))
            y_offset += img.size[1]

    final_img.save(args.out)
    print(f"Saved visualization to {args.out}")

    for img, legends, s_idx in all_images:
        print(f"\n--- Sample {s_idx} ---")
        for legend in legends:
            print(f"  {legend}")


if __name__ == "__main__":
    main()


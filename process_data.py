#!/usr/bin/env python3
"""Convert a published CrossDocked test pickle into an inference LMDB."""
from __future__ import annotations
import argparse
import pickle
import random
import zlib
from pathlib import Path
import lmdb
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from protchord.tokenizer import ProtChordTokenizer

IGNORE_INDEX = -100
UNK_TOKEN_ID = 128257
PROMPTS = (
    "Generate a compound based on the pocket {prot}.",
    "Innovate a compound with the pocket {prot} as a foundation.",
    "Assemble a compound in relation to the pocket {prot}.",
    "Create a compound influenced by the pocket {prot}.",
    "Construct a compound reflecting the essence of the pocket {prot}.",
    "Prepare a compound derived from the principles of the pocket {prot}.",
    "Innovate a compound in the spirit of the pocket {prot}.",
    "Develop a compound that matches the pocket {prot}.",
    "Synthesize a compound derived from the pocket {prot}.",
    "Manufacture a compound using the pocket {prot} as a basis.",
    "Create a compound that corresponds to the pocket {prot}.",
    "Generate a compound that aligns with the pocket {prot}.",
    "Synthesize a compound according to the pocket {prot}.",
    "Craft a compound in the likeness of the pocket {prot}.",
    "Assemble a compound inspired by the essence of the pocket {prot}.",
    "Formulate a compound in accordance with the pocket {prot}.",
    "Fabricate a compound that adheres to the pocket {prot}.",
    "Engineer a compound anchored in the pocket {prot}.",
    "Craft a compound that embodies the pocket {prot}.",
    "Cultivate a compound with the pocket {prot} in mind.",
    "Design a compound that conforms to the pocket {prot}.",
    "Formulate a compound that is influenced by the pocket {prot}.",
    "Produce a compound guided by the pocket {prot}.",
    "Construct a compound modeled on the pocket {prot}.",
    "Design a compound with reference to the pocket {prot}.",
    "Generate a compound reflecting the attributes of the pocket {prot}.",
    "Produce a compound that incorporates the pocket {prot}.",
    "Formulate a compound that mirrors the pocket {prot}.",
    "Fabricate a compound utilizing the pocket {prot}.",
    "Develop a compound that is rooted in the pocket {prot}.",
    "Create a compound that is consistent with the pocket {prot}.",
    "Assemble a compound taking the pocket {prot} into account.",
    "Derive a compound from the characteristics of the pocket {prot}.",
    "Produce a compound based on the criteria of the pocket {prot}.",
    "Compose a compound centered around the pocket {prot}.",
    "Fashion a compound in response to the pocket {prot}.",
    "Invent a compound informed by the pocket {prot}.",
    "Devise a compound inspired by the pocket {prot}.",
    "Construct a compound that reflects the pocket {prot}.",
    "Design a compound following the pocket {prot}.",
    "Develop a compound referencing the pocket {prot}.",
)


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)


def pocket_mask(coords, pocket_coords):
    coords = np.round(np.asarray(coords, dtype=np.float32), 3)
    pocket_coords = np.round(np.asarray(pocket_coords, dtype=np.float32), 3)
    matches = coords[:, None, :] == pocket_coords[None, :, :]
    return np.any(np.all(matches, axis=-1), axis=-1)[..., None]


def make_record(item, index, tokenizer):
    ref_ligand = canonical_smiles(item.get("ref_ligand", item["smiles"]))
    coords = item["coords"]
    protein = "".join(chain for chain in item["aa"] for chain in chain)
    if ref_ligand is None or not coords or not protein or len(protein) > 2048:
        return None
    save_aa = ["".join(chain) for chain in item["aa"]]
    masks = []
    for atom_coord in coords:
        mask = np.zeros((len(atom_coord), 1), dtype=bool)
        for pocket_coord in coords:
            candidate = pocket_mask(atom_coord, pocket_coord)
            if np.any(candidate):
                mask = candidate
                break
        masks.append(mask)
    pocket_ids = tokenizer.encode(f"<protein>{protein}</protein>", prepend_bos=False, add_special_tokens=False)[1:-1]
    pocket_ids = [token for token in pocket_ids if token != UNK_TOKEN_ID]
    prompt = random.choice(PROMPTS).format(prot=f"<protein>{protein}</protein>")
    raw_prompt_ids = tokenizer.encode(prompt, prepend_bos=True, append_eos=False)
    instruct_len = len(raw_prompt_ids)
    tag_ids = tokenizer.encode("<protein>A</protein>", prepend_bos=False, add_special_tokens=False)
    begin, end = tag_ids[0], tag_ids[-1]
    begin_index, end_index = raw_prompt_ids.index(begin), raw_prompt_ids.index(end)
    prompt_ids = raw_prompt_ids[:begin_index] + [begin, -1, end] + raw_prompt_ids[end_index + 1 :]
    masks_count = sum(int(np.count_nonzero(mask)) for mask in masks)
    molecule = Chem.MolFromSmiles(ref_ligand)
    if molecule is None or masks_count == 0 or masks_count != len(pocket_ids):
        return None
    return [2, prompt_ids, [IGNORE_INDEX] * instruct_len, pocket_ids, len(pocket_ids), save_aa, coords,
            (index, item["smiles"], item["meta_info"]),
            (ref_ligand, item["ligand_coords"], molecule.GetNumAtoms()), masks]


def write_lmdb(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), subdir=True, map_size=200 * 1024**3, readonly=False, lock=False, readahead=False)
    with env.begin(write=True) as txn:
        for index, record in enumerate(tqdm(records, desc="writing test LMDB")):
            txn.put(str(index).encode(), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
        metadata = {"keys": list(range(len(records))), "size": len(records)}
        txn.put(b"metadata", zlib.compress(pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)))
    env.sync()
    env.close()
    print(f"wrote {len(records)} records to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-pkl", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer"))
    parser.add_argument("--output-lmdb", type=Path, default=Path("data/crossdocked2020_test.lmdb"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    with open(args.test_pkl, "rb") as handle:
        source = pickle.load(handle)
    tokenizer = ProtChordTokenizer.from_pretrained(args.tokenizer)
    records = [record for index, item in enumerate(tqdm(source, desc="processing test"))
               if (record := make_record(item, index, tokenizer)) is not None]
    write_lmdb(records, args.output_lmdb)


if __name__ == "__main__":
    main()

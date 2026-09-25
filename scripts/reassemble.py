"""Reassemble the chunked TSV files under data/chunks/ into full files under data/full/.

Each original .tsv file was split into <100MB parts (data/chunks/<name>/<name>.part_NNN)
so it could be committed to git without Git LFS. Run this once after cloning:

    python scripts/reassemble.py
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNK_DIR = ROOT / "data" / "chunks"
OUT_DIR = ROOT / "data" / "full"


def reassemble():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in sorted(CHUNK_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        parts = sorted(subdir.glob("*.part_*"))
        if not parts:
            continue
        out_path = OUT_DIR / f"{subdir.name}.tsv"
        print(f"Reassembling {out_path.name} from {len(parts)} parts...")
        with open(out_path, "wb") as out_f:
            for part in parts:
                with open(part, "rb") as in_f:
                    out_f.write(in_f.read())
    print(f"Done. Full TSV files are in {OUT_DIR}")


if __name__ == "__main__":
    reassemble()

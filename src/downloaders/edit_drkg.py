

"""

dataset/vitagraph/vitagraph.tsv
head	interaction	tail	source	type
Gene::NCBI:2157	GENE_BIND	Gene::NCBI:5264	bioarx	Gene-Gene
Gene::NCBI:2157	GENE_BIND	Gene::NCBI:2158	bioarx	Gene-Gene


drkg.tsv
Gene::2157	bioarx::HumGenHumGen:Gene:Gene	Gene::2157
Gene::2157	bioarx::HumGenHumGen:Gene:Gene	Gene::5264
Gene::2157	bioarx::HumGenHumGen:Gene:Gene	Gene::2158

in questo bisgna modifiche aggiungendo i nomi delle colonne e due nuove colonne source e type

Questo è l'aspetto che deve avere il tsv alla fine delle modifiche:
head	interaction	tail	source	type
Gene::2157	bioarx::HumGenHumGen:Gene:Gene	Gene::5264	bioarx	Gene-Gene
Gene::2157	bioarx::HumGenHumGen:Gene:Gene	Gene::2158	bioarx	Gene-Gene

prea la colonna type prendedo il valore della prima parola da head mentre source prendedo la prima parola da interaction


"""

#%%
"""
Transforms drkg.tsv to have columns:

head    interaction  tail    source  type

- source: first token from interaction before '::' (e.g., 'bioarx')
- type: first token from head + '-' + first token from tail (e.g., 'Gene-Gene')

Optionally drops self-loops (head == tail), matching the example.
"""

from __future__ import annotations
import os
import csv
from pathlib import Path

if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')
    
def _first_token(value: str) -> str:
    value = (value or "").strip()
    return value.split("::", 1)[0] if value else ""


def transform_drkg_tsv(
    in_path: str | Path,
    out_path: str | Path | None = None,
    *,
    drop_self_loops: bool = True,
) -> Path:
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path is not None else in_path

    rows_out: list[list[str]] = []
    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue

            # If the file already has a header, skip it
            if row[0].strip().lower() == "head":
                continue

            if len(row) < 3:
                raise ValueError(f"Invalid row (expected 3 columns): {row}")

            head, interaction, tail = (row[0].strip(), row[1].strip(), row[2].strip())

            if drop_self_loops and head == tail:
                continue

            source = _first_token(interaction)              # e.g. bioarx
            type_ = f"{_first_token(head)}-{_first_token(tail)}"  # e.g. Gene-Gene

            rows_out.append([head, interaction, tail, source, type_])

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["head", "interaction", "tail", "source", "type"])
        writer.writerows(rows_out)

    return out_path


if __name__ == "__main__":
    # Overwrites drkg.tsv in place (same file), dropping self-loops.
    transform_drkg_tsv(Path("../dataset/drkg/drkg.tsv"))
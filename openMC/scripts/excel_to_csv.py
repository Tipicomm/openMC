from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------
# Utils
# ---------------------------
def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):
        if (p / "inbox").is_dir() and (p / "model").is_dir() and (p / "out").is_dir() and (p / "scripts").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path.cwd().resolve()


def read_json_robust(path: Path) -> dict[str, Any]:
    """
    Lecture robuste JSON: utf-8 puis fallback utf-8-sig (BOM).
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_for_detection(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return s


def apply_normalize_steps(values: pd.Series, steps: list[str]) -> pd.Series:
    """
    Applique normalize steps refs/fr/*.json : strip/remove_spaces/upper/digits_only/zfill:n
    """
    s = values.astype(str)

    for step in steps or []:
        if step == "strip":
            s = s.str.strip()
        elif step == "remove_spaces":
            s = s.str.replace(r"\s+", "", regex=True)
        elif step == "upper":
            s = s.str.upper()
        elif step == "digits_only":
            s = s.str.replace(r"\D+", "", regex=True)
        elif isinstance(step, str) and step.startswith("zfill:"):
            n = int(step.split(":", 1)[1])
            # Ne pas appliquer zfill sur valeurs vides
            s = s.where(s.isna() | (s == ""), s.str.zfill(n))
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return s


def load_ref(project_root: Path, ref_id: str) -> dict[str, Any] | None:
    """
    ref_id: "fr:cog_commune_insee"
    -> refs/fr/cog_commune_insee.json
    """
    if not ref_id or ":" not in ref_id:
        return None
    ns, name = ref_id.split(":", 1)
    if ns != "fr":
        return None
    p = project_root / "refs" / "fr" / f"{name}.json"
    if not p.exists():
        return None
    return read_json_robust(p)


# ---------------------------
# Numeric parsing & formatting
# ---------------------------
def parse_number_fr(v: str) -> float | None:
    """
    Accepte:
      - "123"
      - "123,4"
      - "1 234,56"
      - "1 234,56" (nbsp)
      - "123.4"
    """
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    s = s.replace("\u00A0", " ").replace("\u202F", " ")
    s = s.replace(" ", "")

    # si "12,34" -> point
    if "," in s and "." not in s:
        s = s.replace(",", ".")

    if not re.match(r"^[+-]?\d+(\.\d+)?$", s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_percentage(v: str) -> float | None:
    """
    Retourne une valeur 0..100 (lisible).
    Cas gérés:
      - "10%" -> 10.0
      - "10,5%" -> 10.5
      - "0.104" (Excel stocke un % en fraction) -> 10.4
    """
    if v is None:
        return None
    s = str(v).strip().replace("\u00A0", " ").replace("\u202F", " ")
    if s == "":
        return None

    if s.endswith("%") or re.match(r"^[+-]?\d+(?:[.,]\d+)?\s*%$", s):
        s2 = s.replace("%", "").strip()
        return parse_number_fr(s2)

    n = parse_number_fr(s)
    if n is None:
        return None
    if 0.0 <= n <= 1.0:
        return n * 100.0
    return n


def format_smart_number(x: float, decimal_sep: str, max_decimals: int = 6) -> str:
    """
    - évite 21.600000 -> 21.6
    - évite 10.000000 -> 10
    - garde jusqu'à max_decimals si nécessaire
    """
    if x is None:
        return ""
    s = f"{x:.{max_decimals}f}"
    s = s.rstrip("0").rstrip(".")
    if decimal_sep == ",":
        s = s.replace(".", ",")
    return s


# ---------------------------
# Main conversion
# ---------------------------
def main(model_path: str):
    project_root = find_project_root(Path.cwd())
    model_file = Path(model_path).expanduser()
    if not model_file.is_absolute():
        model_file = (project_root / model_file).resolve()
    if not model_file.exists():
        raise SystemExit(f"Model introuvable: {model_file}")

    model = read_json_robust(model_file)

    source = model["source"]
    source_file = Path(source)
    if not source_file.is_absolute():
        source_file = (project_root / source_file).resolve()
    if not source_file.exists():
        raise SystemExit(f"Source introuvable: {source_file}")

    sheet = model.get("sheet")
    sheet_to_read = 0 if sheet is None else sheet
    df = pd.read_excel(source_file, sheet_name=sheet_to_read, dtype=str, engine="openpyxl")

    dialect = model.get("dialect", {}) or {}
    delimiter = dialect.get("delimiter", ",")
    encoding = dialect.get("encoding", "utf-8")
    line_terminator = dialect.get("lineTerminator", "\n")

    # data.gouv recommande delimiter="," ; si tu gardes ";", on met décimal=","
    decimal_sep = "." if delimiter == "," else ","

    fields: list[dict[str, Any]] = model["fields"]
    out_cols: dict[str, pd.Series] = {}

    for f in fields:
        source_name = f["source_name"]
        out_name = f["name"]
        semantic = f.get("semantic_type", "text")
        reference = f.get("reference")
        role = f.get("role")  # "code" | "label" | "measure" | "dimension" | ...

        if source_name not in df.columns:
            out_cols[out_name] = pd.Series([pd.NA] * len(df), dtype="object")
            continue

        col = df[source_name].astype(str)

        # 1) REF normalization
        # IMPORTANT: on n'applique les normalize steps QUE sur les colonnes role=="code"
        # sinon, les labels ("Nom_commune") partent en digits_only -> "" -> zfill -> "00000"
        if reference and role == "code":
            ref = load_ref(project_root, reference)
            if ref:
                steps = ref.get("normalize") or []
                col = apply_normalize_steps(col, steps)

        # 2) Semantic conversions
        if semantic == "year":
            # year: on laisse lisible mais propre (pas float)
            s = normalize_for_detection(col)
            out = col.copy()
            out.loc[s.index] = s.str.replace(r"\D+", "", regex=True)
            out_cols[out_name] = out.astype("object")
            continue

        if semantic == "integer":
            vals = col.map(parse_number_fr)
            out_cols[out_name] = vals.map(lambda x: "" if x is None else str(int(round(x)))).astype("object")
            continue

        if semantic == "number":
            vals = col.map(parse_number_fr)
            out_cols[out_name] = vals.map(
                lambda x: "" if x is None else format_smart_number(float(x), decimal_sep, max_decimals=6)
            ).astype("object")
            continue

        if semantic == "percentage":
            vals = col.map(parse_percentage)
            out_cols[out_name] = vals.map(
                lambda x: "" if x is None else format_smart_number(float(x), decimal_sep, max_decimals=3)
            ).astype("object")
            continue

        # par défaut text
        out_cols[out_name] = col.astype("object")

    out_df = pd.DataFrame(out_cols)

    out_dir = project_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{Path(source_file).stem}.clean.csv"

    # On écrit tout en string => pas de perte de zéros, pas de float_format forcé
    out_df.to_csv(
        out_csv,
        index=False,
        sep=delimiter,
        encoding=encoding,
        lineterminator=line_terminator,
        quoting=csv.QUOTE_MINIMAL,
    )

    print(f"OK - CSV généré: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/excel_to_csv.py model/fichier.model.json")
    main(sys.argv[1])

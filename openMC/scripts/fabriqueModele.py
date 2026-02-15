from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
from unidecode import unidecode


# ---------------------------
# Helpers
# ---------------------------
def to_safe_col(name: str) -> str:
    s = unidecode(str(name)).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_") or "col"

    b = s.encode("utf-8")
    if len(b) <= 63:
        return s

    truncated = b[:63]
    while True:
        try:
            return truncated.decode("utf-8").rstrip("_")
        except UnicodeDecodeError:
            truncated = truncated[:-1]


def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):
        if (p / "inbox").is_dir() and (p / "model").is_dir() and (p / "out").is_dir() and (p / "scripts").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path.cwd().resolve()


def read_first_sheet(in_file: Path, sheet: str | None) -> Tuple[pd.DataFrame, str]:
    """
    Lit la 1ère feuille par défaut (évite pandas qui renvoie un dict quand sheet_name=None).
    Retourne (df, sheet_name)
    """
    sheet_to_read = 0 if sheet is None else sheet
    df = pd.read_excel(in_file, sheet_name=sheet_to_read, dtype=str, engine="openpyxl")
    if sheet is None:
        sheet = pd.ExcelFile(in_file, engine="openpyxl").sheet_names[0]
    return df, sheet


# ---------------------------
# Refs
# ---------------------------
def load_refs(project_root: Path) -> list[dict[str, Any]]:
    refs_dir = project_root / "refs" / "fr"
    refs: list[dict[str, Any]] = []
    if not refs_dir.exists():
        return refs
    for p in sorted(refs_dir.glob("*.json")):
        refs.append(json.loads(p.read_text(encoding="utf-8")))
    return refs


def norm(s: str) -> str:
    return unidecode(s or "").lower()


def label_score(col_name: str, keywords: list[str]) -> float:
    """
    Score basé sur la présence de mots-clés dans le header.
    - 0.0 : aucun match
    - 0.5 : 1 match
    - 1.0 : >=2 matches
    """
    if not keywords:
        return 0.0
    s = norm(col_name)
    hits = 0
    for kw in keywords:
        kw_norm = norm(kw)
        if kw_norm and kw_norm in s:
            hits += 1
    if hits >= 2:
        return 1.0
    if hits == 1:
        return 0.5
    return 0.0


def normalize_for_detection(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return s


def apply_normalize_steps(values: pd.Series, steps: list[str]) -> pd.Series:
    """
    Applique les steps de normalisation déclarés dans refs/fr/*.json.
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
            s = s.str.zfill(n)

    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return s


def apply_normalize_steps_for_detection(values: pd.Series, steps: list[str]) -> pd.Series:
    """
    Normalisation pour la *détection* :
    on applique strip/remove_spaces/upper/digits_only
    mais on IGNORE zfill:* (sinon faux positifs année->CP, montants->SIREN, etc.)
    """
    filtered = [st for st in (steps or []) if not str(st).startswith("zfill:")]
    return apply_normalize_steps(values, filtered)


def content_score_with_ref(series: pd.Series, ref: dict[str, Any]) -> float:
    """
    Proportion de valeurs non vides qui matchent le pattern après normalisation (détection sans zfill).
    """
    s = normalize_for_detection(series)
    if s.empty:
        return 0.0

    steps = (ref.get("normalize") or [])
    s2 = apply_normalize_steps_for_detection(s, steps).dropna()
    if s2.empty:
        return 0.0

    patt = ((ref.get("validate") or {}).get("pattern"))
    if not patt:
        return 0.0

    try:
        return float(s2.str.match(patt).mean())
    except re.error:
        return 0.0


def semantic_from_ref_id(ref_id: str) -> str:
    # "fr:cog_commune_insee" -> "cog_commune_insee"
    if ":" in ref_id:
        return ref_id.split(":", 1)[1]
    return ref_id


# ---------------------------
# Role detection
# ---------------------------
ROLE_CODE_HINTS = [
    "_code", "code_", " code",
    "num_", "_num", "numero", "n°",
    "id_", "_id",
    "insee", "siren", "siret",
    "cp", "zip",
]
ROLE_LABEL_HINTS = [
    "_nom", "nom_", " nom",
    "libelle", "libellé", "label",
    "intitule", "intitulé",
    "name", "designation", "désignation",
]
MEASURE_HINTS = [
    "euro", "euros", "€",
    "k€", "keur",
    "montant", "depense", "dépense",
    "taux", "ratio", "pourcentage", "%",
    "par_habitant", "habitant", "hab",
    "moyenne", "median", "médiane",
]
DIMENSION_HINTS = ["annee", "année", "year"]


def is_measure_column(col: str) -> bool:
    c = norm(col)
    return any(norm(h) in c for h in MEASURE_HINTS)


def is_dimension_column(col: str) -> bool:
    c = norm(col)
    return any(norm(h) in c for h in DIMENSION_HINTS)


def infer_role(col: str) -> str:
    if is_measure_column(col):
        return "measure"
    if is_dimension_column(col):
        return "dimension"

    c = norm(col)
    code_hit = any(norm(h) in c for h in ROLE_CODE_HINTS)
    label_hit = any(norm(h) in c for h in ROLE_LABEL_HINTS)

    if code_hit and not label_hit:
        return "code"
    if label_hit and not code_hit:
        return "label"
    return "unknown"


# ---------------------------
# Numeric + Percentage detection
# ---------------------------
_INT_RE = re.compile(r"^[+-]?\d+$")
_NUM_RE = re.compile(r"^[+-]?\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d+)?$|^[+-]?\d+(?:[.,]\d+)?$")


def percentage_ratio(series: pd.Series) -> float:
    s = normalize_for_detection(series)
    if s.empty:
        return 0.0
    x = s.astype(str).str.strip().str.replace("\u00A0", " ", regex=False)
    m = x.str.match(r"^[+-]?\d+(?:[.,]\d+)?\s*%$")
    return float(m.mean())


def _numeric_clean(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.str.replace("\u00A0", " ", regex=False)
    x = x[x != ""]
    return x


def integer_ratio(series: pd.Series) -> float:
    s = normalize_for_detection(series)
    if s.empty:
        return 0.0
    x = _numeric_clean(s)
    if x.empty:
        return 0.0
    return float(x.apply(lambda v: bool(_INT_RE.match(v))).mean())


def number_ratio(series: pd.Series) -> float:
    s = normalize_for_detection(series)
    if s.empty:
        return 0.0
    x = _numeric_clean(s)
    if x.empty:
        return 0.0
    return float(x.apply(lambda v: bool(_NUM_RE.match(v))).mean())


def looks_like_numeric(series: pd.Series, threshold: float = 0.95) -> Tuple[bool, str, float]:
    """
    semantic_type: percentage | integer | number | text
    """
    pr = percentage_ratio(series)
    if pr >= threshold:
        return True, "percentage", float(min(1.0, pr))

    ir = integer_ratio(series)
    if ir >= threshold:
        return True, "integer", float(min(1.0, ir))

    nr = number_ratio(series)
    if nr >= threshold:
        return True, "number", float(min(1.0, nr))

    return False, "text", 0.0


# ---------------------------
# Entity hints (label-only columns)
# ---------------------------
ENTITY_HINTS: dict[str, list[str]] = {
    "fr:cog_commune_insee": ["commune", "ville", "code_insee", "insee", "codgeo"],
    "fr:cog_departement_insee": ["departement", "département", "department", "dep"],
    "fr:cog_region_insee": ["region", "région", "reg"],
    "fr:code_postal": ["code_postal", "cp", "postal", "zip", "zipcode"],
    "fr:siren": ["siren"],
    "fr:siret": ["siret"],
}


def choose_best_ref_for_code(col: str, series: pd.Series, refs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    best = None
    best_score = 0.0
    col_norm = norm(col)

    for r in refs:
        ref_id = r.get("id")
        if not isinstance(ref_id, str):
            continue

        if is_measure_column(col) and (ref_id.startswith("fr:cog_") or ref_id in ("fr:code_postal", "fr:siren", "fr:siret")):
            continue

        detect = r.get("detect", {}) or {}
        keywords = detect.get("label_keywords", []) or []
        neg = detect.get("negative_keywords", []) or []

        patt = (r.get("validate", {}) or {}).get("pattern")
        if not patt:
            continue

        has_neg = any(norm(w) in col_norm for w in neg if w)
        neg_penalty = 0.4 if has_neg else 1.0

        s_field = content_score_with_ref(series, r)
        s_label = label_score(col, keywords)

        score = (s_field * (1.0 + (s_label / 2.0))) * neg_penalty

        if score > best_score:
            best_score = score
            best = r

    if best is None or best_score < 0.85:
        return None

    return {"ref": best, "score": float(min(1.0, best_score))}


def choose_ref_for_label(col: str, refs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    best_ref = None
    best_score = 0.0

    for r in refs:
        ref_id = r.get("id")
        if not isinstance(ref_id, str):
            continue

        hints = ENTITY_HINTS.get(ref_id, [])
        detect = r.get("detect", {}) or {}
        keywords = (detect.get("label_keywords", []) or []) + hints

        s_label = label_score(col, keywords)
        if s_label <= 0.0:
            continue

        score = 0.6 if s_label == 0.5 else 0.75
        if score > best_score:
            best_score = score
            best_ref = r

    if best_ref is None:
        return None

    return {"ref": best_ref, "score": float(best_score)}


# ---------------------------
# Naming rules: add _pct
# ---------------------------
PCT_NAME_HINTS = ["%", "pourcentage", "percentage", "taux", "pct"]


def should_suffix_pct(col_label: str, semantic_type: str) -> bool:
    if semantic_type == "percentage":
        return True
    c = norm(col_label)
    return any(norm(h) in c for h in PCT_NAME_HINTS)


def make_unique(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    i = 2
    while f"{name}_{i}" in used:
        i += 1
    unique = f"{name}_{i}"
    used.add(unique)
    return unique


# ---------------------------
# Main
# ---------------------------
def main(xlsx_path: str, sheet: str | None = None):
    project_root = find_project_root(Path.cwd())

    in_file = Path(xlsx_path).expanduser()
    if not in_file.is_absolute():
        in_file = (project_root / in_file).resolve()
    if not in_file.exists():
        raise SystemExit(f"Fichier introuvable: {in_file}")

    model_dir = project_root / "model"
    out_dir = project_root / "out"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = load_refs(project_root)
    df, sheet = read_first_sheet(in_file, sheet)

    fields: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for col in df.columns:
        col_str = str(col)
        safe_base = to_safe_col(col_str)

        role = infer_role(col_str)
        is_num, num_type, num_conf = looks_like_numeric(df[col], threshold=0.95)

        chosen = None
        proposed_sem = "text"
        reference = None
        confidence = 0.0

        # percent by content
        if is_num and num_type == "percentage":
            proposed_sem = "percentage"
            role = "measure"
            confidence = num_conf

        # measure numeric
        elif role == "measure" and is_num:
            proposed_sem = num_type
            confidence = num_conf

        else:
            # refs for codes
            if role in ("code", "unknown"):
                chosen = choose_best_ref_for_code(col_str, df[col], refs)

            # labels
            if chosen is None and role == "label":
                chosen = choose_ref_for_label(col_str, refs)

            if chosen:
                ref = chosen["ref"]
                ref_id = ref.get("id")
                reference = ref_id if isinstance(ref_id, str) else None
                confidence = float(chosen["score"])

                if role == "label":
                    proposed_sem = "text"
                else:
                    proposed_sem = semantic_from_ref_id(reference) if reference else "text"

            else:
                # fallback year
                s = normalize_for_detection(df[col])
                digits = s.str.replace(r"\D+", "", regex=True)
                p4 = (digits.str.len() == 4).mean() if not s.empty else 0.0

                if p4 >= 0.90 and is_dimension_column(col_str):
                    proposed_sem = "year"
                    role = "dimension"
                    confidence = float(min(1.0, p4))
                elif is_num:
                    proposed_sem = num_type
                    confidence = num_conf
                    if role == "unknown":
                        role = "measure"
                else:
                    proposed_sem = "text"
                    confidence = 0.0

        # ---- apply naming rule _pct AFTER semantic is known
        out_name = safe_base
        if should_suffix_pct(col_str, proposed_sem) and not out_name.endswith("_pct"):
            out_name = f"{out_name}_pct"

        out_name = make_unique(out_name, used_names)

        fields.append(
            {
                "source_name": col_str,
                "name": out_name,
                "label": col_str,
                "semantic_type": proposed_sem,
                "reference": reference,
                "role": role,
                "nullable": True,
                "notes": {"confidence": confidence},
            }
        )

    try:
        source_rel = in_file.relative_to(project_root)
        source_str = str(source_rel)
    except ValueError:
        source_str = str(in_file)

    model = {
        "source": source_str,
        "sheet": sheet,
        "dialect": {"delimiter": ";", "encoding": "utf-8", "lineTerminator": "\n"},
        "options": {"invalid_policy": "blank_and_report", "normalize_headers": True},
        "fields": fields,
    }

    out_model = model_dir / f"{in_file.stem}.model.json"
    out_model.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    preview = {
        "file": in_file.name,
        "sheet": sheet,
        "columns": len(df.columns),
        "rows": int(len(df)),
        "proposals": [
            {
                "name": f["name"],
                "semantic_type": f["semantic_type"],
                "reference": f.get("reference"),
                "role": f.get("role"),
                "confidence": f["notes"]["confidence"],
            }
            for f in fields
        ],
    }
    (out_dir / f"{in_file.stem}.preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"OK - modèle généré: {out_model}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/fabriqueModele.py inbox/fichier.xlsx [nom_feuille]")
    xlsx = sys.argv[1]
    sheet = sys.argv[2] if len(sys.argv) >= 3 else None
    main(xlsx, sheet)

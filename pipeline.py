import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import pyreadstat


"""
Metadata-driven pipeline:

- Read responses_schema.json to derive:
  - column names
  - pandas dtypes
  - datetime columns
  - value labels for singleChoice fields

- Read responses_data.json into a Pandas DataFrame and apply schema-driven types.

- Export the DataFrame to an SPSS .sav file with value labels.
"""


# ----------------------------- Schema helpers ----------------------------- #

def map_fieldtype_to_dtype(field_type: str) -> str:
    """
    Map schema fieldType to a pandas dtype string.

    Uses nullable integer (Int64) so missing values are supported.
    """
    ft = (field_type or "").lower()
    if ft == "numeric":
        return "Int64"
    if ft in ("text", "singlechoice"):
        return "string"
    if ft == "datetime":
        return "datetime64[ns]"
    return "string"


def extract_schema_info(
    schema_path: Path,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, str], List[str]]:
    """
    Read responses_schema.json and return:
        - names: ordered list of all variable names (keys + fields)
        - fields: list of field dicts
        - column_dtypes: {column_name: pandas_dtype} for non-datetime columns
        - datetime_columns: [column_name, ...]
    """
    with open(schema_path, "r", encoding="utf-8") as fh:
        j = json.load(fh)

    schema = j.get("data", {}).get("schema") if isinstance(j, dict) else None
    if not isinstance(schema, dict):
        raise ValueError("Schema JSON must contain 'data.schema' as an object.")

    names: List[str] = []
    fields: List[Dict[str, Any]] = []
    column_dtypes_local: Dict[str, str] = {}
    datetime_cols_local: List[str] = []

    # Keys (e.g. responseid)
    for k in schema.get("keys", []) or []:
        if not isinstance(k, dict):
            continue
        fname = k.get("name")
        if not fname:
            continue
        names.append(fname)
        pd_dtype = map_fieldtype_to_dtype(k.get("fieldType", "text") or "text")
        if pd_dtype == "datetime64[ns]":
            datetime_cols_local.append(fname)
        else:
            column_dtypes_local[fname] = pd_dtype

    # Fields
    fields = schema.get("fields") or []
    for f in fields:
        if not isinstance(f, dict):
            continue
        fname = f.get("name")
        if not fname:
            continue
        names.append(fname)
        ftype = f.get("fieldType", "text")
        pd_dtype = map_fieldtype_to_dtype(ftype)
        if pd_dtype == "datetime64[ns]":
            datetime_cols_local.append(fname)
        else:
            column_dtypes_local[fname] = pd_dtype

    return names, fields, column_dtypes_local, datetime_cols_local


def build_value_labels(fields: List[Dict[str, Any]]) -> Dict[str, Dict[Any, str]]:
    """
    Build SPSS-style value labels from schema fields.

    Returns:
        {column_name: {code: label_text}}
    """
    value_labels: Dict[str, Dict[Any, str]] = {}

    for f in fields:
        if not isinstance(f, dict):
            continue

        if f.get("fieldType") != "singleChoice":
            continue

        col_name = f.get("name")
        if not col_name:
            continue

        options = f.get("options") or []
        col_labels: Dict[Any, str] = {}

        for opt in options:
            code_raw = opt.get("code")
            code_val: Any = code_raw  # keep as string; your codes are often strings

            texts = opt.get("texts") or []
            if texts:
                label_text = texts[0].get("text", str(code_raw))
            else:
                label_text = str(code_raw)

            col_labels[code_val] = label_text

        if col_labels:
            value_labels[col_name] = col_labels

    return value_labels


# ----------------------------- Data helpers ----------------------------- #

def load_responses_to_df(
    responses_data_path: Path,
    column_dtypes: Dict[str, str],
    datetime_columns: List[str],
) -> pd.DataFrame:
    """
    Load responses_data.json into a DataFrame and apply schema-driven dtypes.
    """
    if not responses_data_path.exists():
        raise FileNotFoundError(f"Responses data file not found: {responses_data_path}")

    with open(responses_data_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("Expected responses_data.json to contain a JSON array.")

    df = pd.DataFrame(data)

    # Apply non‑datetime dtypes
    for col, dt in column_dtypes.items():
        if col in df.columns and dt != "datetime64[ns]":
            df[col] = df[col].astype(dt, errors="ignore")

    # Parse datetime columns
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# ----------------------------- Main pipeline ----------------------------- #

def main() -> None:
    # Configurable base directory and file names
    base_dir = Path("Testdata2024")

    responses_schema_path = base_dir / "responses_schema.json"
    responses_data_path = base_dir / "responses_data.json"
    sav_output_path = base_dir / "responses_data.sav"

    if not responses_schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {responses_schema_path}")

    # 1) Schema → metadata
    schema_names, fields, column_dtypes, datetime_columns = extract_schema_info(
        responses_schema_path
    )
    value_labels = build_value_labels(fields)

    print(f"Found {len(schema_names)} variables in schema.")
    print(f"{len(column_dtypes)} typed columns, {len(datetime_columns)} datetime columns.")

    # 2) Data → DataFrame with types
    df = load_responses_to_df(responses_data_path, column_dtypes, datetime_columns)

    # enforce schema column order
    schema_order = [name for name in schema_names if name in df.columns]
    if schema_order:
        df = df[schema_order]

    # Quick validation prints
    print("DataFrame shape:", df.shape)
    print("DataFrame dtypes (first few):")
    print(df.dtypes.head())

    # simple content check for status if present in schema
    if "status" in df.columns:
        print("Status value counts:")
        print(df["status"].value_counts(dropna=False))

    # 3) DataFrame → SPSS .sav with value labels
    pyreadstat.write_sav(
        df,
        sav_output_path,
        variable_value_labels=value_labels,
    )

    print("Saved .sav to:", sav_output_path)


if __name__ == "__main__":
    main()

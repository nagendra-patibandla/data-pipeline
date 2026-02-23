import json
from pathlib import Path

import pandas as pd

from pipeline import (
    map_fieldtype_to_dtype,
    extract_schema_info,
    build_value_labels,
    load_responses_to_df,
)


def test_map_fieldtype_to_dtype():
    assert map_fieldtype_to_dtype("numeric") == "Int64"
    assert map_fieldtype_to_dtype("text") == "string"
    assert map_fieldtype_to_dtype("singleChoice") == "string"
    assert map_fieldtype_to_dtype("dateTime") == "datetime64[ns]"
    # unknown → string
    assert map_fieldtype_to_dtype("unknown") == "string"


def test_build_value_labels():
    # minimal fake schema fields (similar to your real schema)[file:2]
    fields = [
        {
            "name": "status",
            "fieldType": "singleChoice",
            "options": [
                {
                    "code": "complete",
                    "texts": [{"language": "en", "text": "Complete"}],
                },
                {
                    "code": "incomplete",
                    "texts": [{"language": "en", "text": "Incomplete"}],
                },
            ],
        },
        {
            "name": "age",
            "fieldType": "numeric",
        },
    ]

    value_labels = build_value_labels(fields)

    assert "status" in value_labels
    assert value_labels["status"]["complete"] == "Complete"
    assert value_labels["status"]["incomplete"] == "Incomplete"
    # numeric field should not have labels
    assert "age" not in value_labels


def test_extract_schema_info_and_load_responses(tmp_path: Path):
    """
    Integration-style test on a tiny synthetic schema + responses.
    It checks:
      - schema parsing to dtypes/datetimes
      - loading responses and applying those types
    """
    # 1) Write a minimal schema file (mirrors real structure)[file:2]
    schema_content = {
        "data": {
            "schema": {
                "name": "response",
                "keys": [
                    {"name": "responseid", "fieldType": "numeric"},
                ],
                "fields": [
                    {"name": "respid", "fieldType": "numeric"},
                    {"name": "status", "fieldType": "singleChoice"},
                    {"name": "interview_start", "fieldType": "dateTime"},
                ],
            }
        }
    }
    schema_path = tmp_path / "responses_schema.json"
    schema_path.write_text(json.dumps(schema_content), encoding="utf-8")

    # 2) Create a tiny responses_data.json compatible with that schema
    responses = [
        {
            "responseid": 1,
            "respid": 10,
            "status": "complete",
            "interview_start": "2023-06-22T21:26:47+00:00",
        },
        {
            "responseid": 2,
            "respid": 20,
            "status": "incomplete",
            "interview_start": "2023-06-23T10:00:00+00:00",
        },
    ]
    data_path = tmp_path / "responses_data.json"
    data_path.write_text(json.dumps(responses), encoding="utf-8")

    # 3) Parse schema
    names, fields, column_dtypes, datetime_columns = extract_schema_info(schema_path)

    # Check names and types
    assert names == ["responseid", "respid", "status", "interview_start"]
    assert column_dtypes["responseid"] == "Int64"
    assert column_dtypes["respid"] == "Int64"
    assert column_dtypes["status"] == "string"
    assert "interview_start" in datetime_columns

    # 4) Load responses and apply types
    df = load_responses_to_df(data_path, column_dtypes, datetime_columns)

    # Shape
    assert df.shape == (2, 4)

    # Dtypes
    assert df["responseid"].dtype == "Int64"
    assert df["respid"].dtype == "Int64"
    assert str(df["status"].dtype) == "string"
    assert pd.api.types.is_datetime64_any_dtype(df["interview_start"])

    # Content sanity check
    assert set(df["status"].unique()) == {"complete", "incomplete"}
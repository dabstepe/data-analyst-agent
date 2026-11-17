"""Conversational data analyst: answers questions about a CSV file from Google Drive.

The agent has a fixed set of tools (load, describe, preview, filter, aggregate).
There is no sandbox and no code execution: every number in an answer comes from
one of these tools.
"""

import csv
import io
import re
import statistics
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from mistralai import agents

MAX_CSV_BYTES = 10_000_000
MAX_RETURNED_ROWS = 50
MAX_EXAMPLES = 5

_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}
_GROUPING = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")

_DATASETS: dict[str, dict[str, Any]] = {}


class Condition(BaseModel):
    column: str
    op: Literal[
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "starts_with",
        "ends_with",
        "in",
        "not_in",
        "is_empty",
        "not_empty",
    ]
    value: Union[str, int, float, bool, list[Union[str, int, float]]] = None


class Metric(BaseModel):
    func: Literal["count", "sum", "mean", "median", "min", "max", "nunique"]
    column: str = None


def _number(raw: str) -> float | None:
    text = raw.strip()
    try:
        return float(text)
    except ValueError:
        pass
    if _GROUPING.match(text):
        try:
            return float(text.replace(",", ""))
        except ValueError:
            return None
    return None


def _infer_kinds(rows: list[dict[str, str]], columns: list[str]) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for col in columns:
        numeric = True
        boolean = True
        seen = False
        for row in rows:
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            seen = True
            if numeric and _number(raw) is None:
                numeric = False
            if boolean and raw.lower() not in _TRUE | _FALSE:
                boolean = False
            if not numeric and not boolean:
                break
        if not seen:
            kinds[col] = "empty"
        elif numeric:
            kinds[col] = "number"
        elif boolean:
            kinds[col] = "boolean"
        else:
            kinds[col] = "text"
    return kinds


def _coerce(kind: str, raw: str | None) -> Any:
    text = (raw or "").strip()
    if text == "":
        return None
    if kind == "number":
        return _number(text)
    if kind == "boolean":
        lowered = text.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        return None
    return text


def _get_dataset(name: str) -> dict[str, Any]:
    dataset = _DATASETS.get(name)
    if dataset is None:
        loaded = ", ".join(f"'{n}'" for n in _DATASETS) or "none"
        raise agents.ToolError(
            f"No dataset named '{name}' is loaded in this conversation. "
            f"Datasets loaded: {loaded}. Load a file first with load_csv."
        )
    return dataset


def _check_columns(dataset: dict[str, Any], names: list[str], usage: str) -> None:
    known = set(dataset["columns"])
    for name in names:
        if name not in known:
            raise agents.ToolError(
                f"Column '{name}' does not exist. Available columns: {dataset['columns']}."
            )


def _match(dataset: dict[str, Any], condition: Condition) -> list[bool]:
    col = condition.column
    kind = dataset["kinds"].get(col, "text")
    op = condition.op
    if op in {"is_empty", "not_empty"}:
        if op == "is_empty":
            return [not (row.get(col) or "").strip() for row in dataset["rows"]]
        return [bool((row.get(col) or "").strip()) for row in dataset["rows"]]
    if op in {"in", "not_in"}:
        if not isinstance(condition.value, list):
            raise agents.ToolError(f"Operator '{op}' needs a list of values.")
        wanted = {str(v).strip().lower() for v in condition.value}
        flags = []
        for row in dataset["rows"]:
            raw = (row.get(col) or "").strip().lower()
            hit = raw in wanted
            flags.append(hit if op == "in" else not hit)
        return flags
    if isinstance(condition.value, list):
        raise agents.ToolError(f"Operator '{op}' needs a single value, not a list.")
    if op in {"gt", "gte", "lt", "lte"}:
        if kind != "number":
            raise agents.ToolError(
                f"Operator '{op}' on column '{col}' needs a numeric column, "
                f"but it is {kind}."
            )
        bound = float(condition.value)
        flags = []
        for row in dataset["rows"]:
            cell = _coerce(kind, row.get(col))
            if cell is None:
                flags.append(False)
                continue
            if op == "gt":
                flags.append(cell > bound)
            elif op == "gte":
                flags.append(cell >= bound)
            elif op == "lt":
                flags.append(cell < bound)
            else:
                flags.append(cell <= bound)
        return flags
    value = condition.value
    flags = []
    for row in dataset["rows"]:
        raw = (row.get(col) or "").strip()
        if op == "eq":
            hit = _eq(kind, raw, value)
        elif op == "ne":
            hit = not _eq(kind, raw, value)
        elif op == "contains":
            hit = str(value).lower() in raw.lower()
        elif op == "starts_with":
            hit = raw.lower().startswith(str(value).lower())
        else:
            hit = raw.lower().endswith(str(value).lower())
        flags.append(hit)
    return flags


def _eq(kind: str, raw: str, value: Any) -> bool:
    if kind == "number":
        cell = _number(raw)
        try:
            return cell is not None and cell == float(value)
        except (TypeError, ValueError):
            return False
    if kind == "boolean":
        cell = _coerce(kind, raw)
        if cell is None:
            return False
        if isinstance(value, bool):
            return cell == value
        text = str(value).strip().lower()
        if text in _TRUE:
            return cell is True
        if text in _FALSE:
            return cell is False
        return False
    return raw.lower() == str(value).strip().lower()


def _apply_filters(dataset: dict[str, Any], filters: list[Condition] | None) -> list[dict[str, str]]:
    rows = dataset["rows"]
    if not filters:
        return rows
    kept = [True] * len(rows)
    for condition in filters:
        _check_columns(dataset, [condition.column], "filter")
        for i, hit in enumerate(_match(dataset, condition)):
            if not hit:
                kept[i] = False
    return [row for row, keep in zip(rows, kept) if keep]


@agents.tool(model_access="direct")
def load_csv(
    name: Annotated[str, Field(description="Short name for the dataset, used by every other tool.")],
    csv_content: Annotated[str, Field(description="The raw CSV text, including the header row.")],
) -> dict[str, Any]:
    """Load a CSV file into memory and register it under a name for this conversation."""
    if len(csv_content.encode("utf-8")) > MAX_CSV_BYTES:
        raise agents.ToolError(
            "The CSV is larger than 10 MB, which this tool refuses. "
            "Ask the user for a smaller export of the same data."
        )
    reader = csv.reader(io.StringIO(csv_content))
    try:
        rows_raw = list(reader)
    except csv.Error as error:
        raise agents.ToolError(f"The content is not valid CSV: {error}") from error
    if not rows_raw:
        raise agents.ToolError("The content is empty: no header row found.")
    columns = [column.strip() for column in rows_raw[0]]
    if len(set(columns)) != len(columns):
        raise agents.ToolError(f"The header has duplicate column names: {columns}.")
    data_rows = []
    for values in rows_raw[1:]:
        if not any(value.strip() for value in values):
            continue
        padded = values + [""] * (len(columns) - len(values))
        data_rows.append(dict(zip(columns, (value.strip() for value in padded))))
    if not data_rows:
        raise agents.ToolError("The CSV has a header but no data rows.")
    _DATASETS[name] = {
        "columns": columns,
        "rows": data_rows,
        "kinds": _infer_kinds(data_rows, columns),
    }
    return {
        "dataset": name,
        "rows_loaded": len(data_rows),
        "columns": [
            {"name": column, "kind": _DATASETS[name]["kinds"][column]} for column in columns
        ],
    }


@agents.tool(model_access="direct")
def list_datasets() -> list[dict[str, Any]]:
    """List the datasets loaded in this conversation, with their sizes."""
    return [
        {"name": name, "rows": len(data["rows"]), "columns": data["columns"]}
        for name, data in _DATASETS.items()
    ]


@agents.tool(model_access="direct")
def describe_columns(
    name: Annotated[str, Field(description="Dataset name given to load_csv.")],
) -> list[dict[str, Any]]:
    """Describe every column of a dataset: inferred type, missing values, uniques, range, examples."""
    dataset = _get_dataset(name)
    rows = dataset["rows"]
    descriptions = []
    for column in dataset["columns"]:
        kind = dataset["kinds"][column]
        raw_values = [(row.get(column) or "").strip() for row in rows]
        non_empty = [value for value in raw_values if value != ""]
        unique = {value.lower() for value in non_empty}
        entry: dict[str, Any] = {
            "column": column,
            "kind": kind,
            "rows": len(rows),
            "non_empty": len(non_empty),
            "missing": len(rows) - len(non_empty),
            "unique": len(unique),
        }
        if kind == "number":
            numbers = [_number(value) for value in non_empty]
            numbers = [value for value in numbers if value is not None]
            if numbers:
                entry["min"] = min(numbers)
                entry["max"] = max(numbers)
                entry["mean"] = round(statistics.fmean(numbers), 4)
        elif kind == "boolean":
            truths = sum(1 for value in non_empty if value.lower() in _TRUE)
            entry["true"] = truths
            entry["false"] = len(non_empty) - truths
        else:
            entry["examples"] = sorted(unique)[:MAX_EXAMPLES]
        descriptions.append(entry)
    return descriptions


@agents.tool(model_access="direct")
def preview_rows(
    name: Annotated[str, Field(description="Dataset name given to load_csv.")],
    limit: Annotated[int, Field(ge=1, le=MAX_RETURNED_ROWS)] = 5,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> dict[str, Any]:
    """Show a few rows of the dataset as-is, useful to see real values after loading."""
    dataset = _get_dataset(name)
    rows = dataset["rows"][offset : offset + limit]
    return {"total_rows": len(dataset["rows"]), "returned": len(rows), "rows": rows}


@agents.tool(model_access="direct")
def filter_rows(
    name: Annotated[str, Field(description="Dataset name given to load_csv.")],
    conditions: Annotated[list[Condition], Field(description="All conditions must hold (AND).")],
    limit: Annotated[int, Field(ge=1, le=MAX_RETURNED_ROWS)] = 20,
) -> dict[str, Any]:
    """Return the rows matching all conditions, with the total count of matches."""
    dataset = _get_dataset(name)
    matched = _apply_filters(dataset, conditions)
    return {
        "matching_rows": len(matched),
        "total_rows": len(dataset["rows"]),
        "returned": min(len(matched), limit),
        "rows": matched[:limit],
    }


@agents.tool(model_access="direct")
def aggregate(
    name: Annotated[str, Field(description="Dataset name given to load_csv.")],
    group_by: Annotated[list[str], Field(description="Columns to group by; empty list for one grand total row.")],
    metrics: Annotated[list[Metric], Field(description="Aggregations to compute per group.")],
    filters: Annotated[list[Condition] | None, Field(description="Optional conditions applied before grouping.")] = None,
    sort_by: Annotated[str | None, Field(description="Metric label to sort by, e.g. 'sum_revenue'.")] = None,
    descending: bool = True,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Group rows and compute aggregates (count, sum, mean, median, min, max, nunique) per group."""
    dataset = _get_dataset(name)
    _check_columns(dataset, group_by, "group_by")
    for metric in metrics:
        if metric.func != "count":
            _check_columns(dataset, [metric.column], f"metric {metric.func}")
    rows = _apply_filters(dataset, filters)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple((row.get(column) or "").strip() or "(missing)" for column in group_by)
        groups.setdefault(key, []).append(row)
    labels = [f"{metric.func}_{metric.column}" if metric.func != "count" else "count" for metric in metrics]
    computed = []
    for key, group_rows in groups.items():
        entry: dict[str, Any] = dict(zip(group_by, key))
        for metric, label in zip(metrics, labels):
            if metric.func == "count":
                entry[label] = len(group_rows)
                continue
            kind = dataset["kinds"].get(metric.column, "text")
            values = [_coerce(kind, row.get(metric.column)) for row in group_rows]
            values = [value for value in values if value is not None]
            if metric.func == "nunique":
                entry[label] = len({str(value) for value in values})
            elif metric.func in {"sum", "mean", "median"}:
                if kind != "number":
                    raise agents.ToolError(
                        f"'{metric.func}' needs a numeric column, but '{metric.column}' is {kind}."
                    )
                numbers = [float(value) for value in values]
                if not numbers:
                    entry[label] = None
                elif metric.func == "sum":
                    entry[label] = round(sum(numbers), 4)
                elif metric.func == "mean":
                    entry[label] = round(statistics.fmean(numbers), 4)
                else:
                    entry[label] = round(statistics.median(numbers), 4)
            else:
                if not values:
                    entry[label] = None
                elif kind == "number":
                    numbers = [float(value) for value in values]
                    entry[label] = min(numbers) if metric.func == "min" else max(numbers)
                else:
                    entry[label] = min(values) if metric.func == "min" else max(values)
        computed.append(entry)
    if sort_by:
        if sort_by not in labels:
            raise agents.ToolError(f"sort_by must be one of the computed metrics: {labels}.")
        computed.sort(
            key=lambda entry: (entry[sort_by] is None, entry[sort_by]),
            reverse=descending,
        )
    return {
        "rows_aggregated": len(rows),
        "total_groups": len(computed),
        "returned_groups": min(len(computed), limit),
        "groups": computed[:limit],
    }


agent = agents.Agent(
    model="mistral-large-4",
    name="data-analyst",
    instructions=(
        "You are a conversational data analyst. The user gives you a CSV file that lives in their "
        "Google Drive (a link or a file name) and asks questions about it in plain language.\n"
        "\n"
        "Getting the data:\n"
        "- Use the Google Drive connector tools to find the file (search by name when the user "
        "gives a name, resolve the id from a link otherwise) and download its content. If the file "
        "is a Google Sheet, export it as CSV.\n"
        "- Pass the downloaded CSV text to load_csv under a short dataset name. Then call "
        "describe_columns and preview_rows once to learn the schema before answering.\n"
        "- If the download fails or returns something that is not CSV text, say exactly what "
        "happened and ask the user to check the file or share it as CSV.\n"
        "\n"
        "Answering questions:\n"
        "- Base every number on a tool result. Never compute totals, averages or counts yourself "
        "and never invent values: you have no calculator and no code execution.\n"
        "- Use aggregate for 'how many', 'average', 'sum', 'top N by', 'broken down by' questions; "
        "use filter_rows for subsets and to show matching rows; use describe_columns to talk about "
        "data quality, ranges and distributions.\n"
        "- Quote the concrete figures and keep small tables when they help. State the row count "
        "your answer is based on.\n"
        "- If a question needs a column that does not exist or values that are messy (mixed types, "
        "many missing values), say so and suggest the closest available analysis.\n"
        "\n"
        "Memory:\n"
        "- The user should never repeat the link: remember the active dataset for the whole "
        "conversation and use it for follow-ups. list_datasets shows what is loaded; when the user "
        "provides another file, load it under a new name and say which dataset is now active.\n"
        "- Answer in the language the user writes in."
    ),
    tools=[load_csv, list_datasets, describe_columns, preview_rows, filter_rows, aggregate],
    connectors=[agents.connector("google_drive_mcp")],
)

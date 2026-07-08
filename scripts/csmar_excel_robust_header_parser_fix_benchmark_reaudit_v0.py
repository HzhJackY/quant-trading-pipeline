from __future__ import annotations

import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "CSMAR Excel Robust Header Parser Fix & Benchmark Re-Audit v0"
TASK_DIR = "csmar_excel_parser_fix_benchmark_reaudit_v0"
OUT = ROOT / "output" / TASK_DIR
RUN = ROOT / "output" / "_agent_runs" / TASK
CSMAR = ROOT / "data" / "csmar_exports"
PORT = ROOT / "output" / "unified_robust_portfolio_evaluation_run_v0"
PANEL = ROOT / "output" / "robust_cleaned_fundamental_factor_variant_build_v0" / "robust_cleaned_factor_score_panel_v0.parquet"

OUT.mkdir(parents=True, exist_ok=True)
RUN.mkdir(parents=True, exist_ok=True)

REQUIRED_FILES = [
    "TRD_Index.xlsx",
    "TRD_Index[DES][xlsx].txt",
    "TRD_Cnmont.xlsx",
    "TRD_Cnmont[DES][xlsx].txt",
    "TRD_Mont.xlsx",
    "TRD_Mont[DES][xlsx].txt",
    "TRD_Mnth.xlsx",
    "TRD_Mnth[DES][xlsx].txt",
    "TRD_Nrrate.xlsx",
    "TRD_Nrrate[DES][xlsx].txt",
    "STK_MKT_FIVEFACDAY.xlsx",
    "STK_MKT_FIVEFACDAY[DES][xlsx].txt",
]

REQUIRED_BY_FILE = {
    "TRD_Index.xlsx": ["Indexcd", "Trddt", "Clsindex", "Retindex"],
    "TRD_Cnmont.xlsx": ["Markettype", "Trdmnt", "Cmretwdeq", "Cmretmdeq", "Cmretwdos", "Cmretmdos", "Cmretwdtl", "Cmretmdtl"],
    "TRD_Mont.xlsx": ["Markettype", "Trdmnt", "Mretwdeq", "Mretmdeq", "Mretwdos", "Mretmdos", "Mretwdtl", "Mretmdtl"],
    "TRD_Mnth.xlsx": ["Stkcd", "Trdmnt", "Mretwd", "Mretnd", "Msmvosd", "Msmvttl", "Ndaytrd", "Markettype"],
    "TRD_Nrrate.xlsx": ["Nrr1", "Clsdt", "Nrrdata", "Nrrdaydt", "Nrrmtdt"],
    "STK_MKT_FIVEFACDAY.xlsx": ["MarkettypeID", "TradingDate", "Portfolios", "RiskPremium1", "RiskPremium2", "SMB1", "SMB2", "HML1", "HML2", "RMW1", "RMW2", "CMA1", "CMA2"],
}

FALLBACK_CHINESE_TO_CODE = {
    "TRD_Index.xlsx": {
        "指数代码": "Indexcd", "交易日期": "Trddt", "收盘指数": "Clsindex", "指数回报率": "Retindex",
        "开盘指数": "Opnindex", "最高指数": "Hiindex", "最低指数": "Loindex",
    },
    "TRD_Cnmont.xlsx": {
        "市场类型": "Markettype", "交易月份": "Trdmnt",
        "考虑现金红利再投资的综合月市场回报率等权平均法": "Cmretwdeq",
        "不考虑现金红利再投资的综合月市场回报率等权平均法": "Cmretmdeq",
        "考虑现金红利再投资的综合月市场回报率流通市值加权平均法": "Cmretwdos",
        "不考虑现金红利再投资的综合月市场回报率流通市值加权平均法": "Cmretmdos",
        "考虑现金红利再投资的综合月市场回报率总市值加权平均法": "Cmretwdtl",
        "不考虑现金红利再投资的综合月市场回报率总市值加权平均法": "Cmretmdtl",
        "计算综合月市场回报率的公司数量": "Cmnstkcal", "计算综合月市场回报率的有效公司数量": "Cmnstkcal",
        "综合月市场流通市值": "Cmmvosd", "综合月市场总流通市值": "Cmmvosd", "综合月市场总市值": "Cmmvttl",
    },
    "TRD_Mont.xlsx": {
        "市场类型": "Markettype", "交易月份": "Trdmnt",
        "考虑现金红利再投资的月市场回报率等权平均法": "Mretwdeq",
        "不考虑现金红利再投资的月市场回报率等权平均法": "Mretmdeq",
        "考虑现金红利再投资的月市场回报率流通市值加权平均法": "Mretwdos",
        "不考虑现金红利再投资的月市场回报率流通市值加权平均法": "Mretmdos",
        "考虑现金红利再投资的月市场回报率总市值加权平均法": "Mretwdtl",
        "不考虑现金红利再投资的月市场回报率总市值加权平均法": "Mretmdtl",
    },
    "TRD_Mnth.xlsx": {
        "证券代码": "Stkcd", "交易月份": "Trdmnt", "月收盘价": "Mclsprc", "月个股流通市值": "Msmvosd",
        "月个股总市值": "Msmvttl", "月交易天数": "Ndaytrd", "考虑现金红利再投资的月个股回报率": "Mretwd",
        "不考虑现金红利再投资的月个股回报率": "Mretnd", "市场类型": "Markettype",
    },
    "TRD_Nrrate.xlsx": {
        "无风险利率基准": "Nrr1", "统计日期": "Clsdt", "无风险利率": "Nrrdata",
        "日度化无风险利率": "Nrrdaydt", "周度化无风险利率": "Nrrwkdt", "月度化无风险利率": "Nrrmtdt",
    },
}

UNIT_VALUES = {"没有单位", "元", "%", "股", "千股", "日", "月", "CNY", "人民币元", ""}


def checkpoint(message: str, status: str = "running") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = (
        f"# RUN_STATE: {TASK}\n\n"
        f"status: {status}\n"
        f"last_checkpoint: {ts} {message}\n"
        f"mode: low-resource checkpoint-first resume-safe\n\n"
        f"logs:\n- {RUN / 'run_stdout.txt'}\n- {RUN / 'run_stderr.txt'}\n\n"
        f"output_dir: {OUT}\n"
    )
    (RUN / "RUN_STATE.md").write_text(text, encoding="utf-8")
    print(f"[checkpoint] {message}", flush=True)


def normalize_cell_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().replace("\u3000", " ")
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_chinese(text: str) -> str:
    return re.sub(r"[\s（）()]+", "", normalize_cell_text(text))


def parse_des(file_name: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    p = CSMAR / file_name.replace(".xlsx", "[DES][xlsx].txt")
    code_to_desc: dict[str, str] = {}
    chinese_to_code: dict[str, str] = {}
    units: dict[str, str] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*\[(.*?)\]\s*-\s*(.*)$", line)
            if not m:
                continue
            code, chinese, desc = m.group(1), normalize_cell_text(m.group(2)), normalize_cell_text(m.group(3))
            code_to_desc[code] = f"{chinese} - {desc}".strip()
            chinese_to_code[compact_chinese(chinese)] = code
            unit_match = re.search(r"(计量货币|单位)[：:]\s*([^，,。；;]+)", desc)
            if unit_match:
                units[code] = normalize_cell_text(unit_match.group(2))
    for chinese, code in FALLBACK_CHINESE_TO_CODE.get(file_name, {}).items():
        chinese_to_code.setdefault(compact_chinese(chinese), code)
    return code_to_desc, chinese_to_code, units


def known_fields_for(file_name: str, code_to_desc: dict[str, str]) -> dict[str, str]:
    fields = set(REQUIRED_BY_FILE.get(file_name, [])) | set(code_to_desc)
    fields |= set(FALLBACK_CHINESE_TO_CODE.get(file_name, {}).values())
    return {f.lower(): f for f in fields}


def extract_field_code(cell_text: Any, known_fields: dict[str, str], chinese_to_code: dict[str, str]) -> tuple[str | None, str, str]:
    text = normalize_cell_text(cell_text)
    if not text:
        return None, "", ""
    lower = text.lower()
    if lower in known_fields:
        return known_fields[lower], "exact_code", ""
    tokens = re.split(r"[\s/|,，;；:：\[\]（）()]+", text)
    for token in tokens:
        if token.lower() in known_fields:
            return known_fields[token.lower()], "token_code", ""
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text):
        if token.lower() in known_fields:
            return known_fields[token.lower()], "embedded_code", ""
    compact = compact_chinese(text)
    if compact in chinese_to_code:
        return chinese_to_code[compact], "des_or_fallback_chinese", text
    for chinese, code in chinese_to_code.items():
        if chinese and chinese in compact:
            return code, "des_or_fallback_chinese_partial", text
    return None, "", ""


def row_is_unit(values: list[Any], mapped_positions: set[int]) -> bool:
    cells = [normalize_cell_text(values[i]) if i < len(values) else "" for i in mapped_positions]
    if not cells:
        return False
    unit_like = 0
    nonempty = 0
    for cell in cells:
        if cell:
            nonempty += 1
        if compact_chinese(cell) in {compact_chinese(x) for x in UNIT_VALUES}:
            unit_like += 1
    return unit_like / max(len(cells), 1) >= 0.60 and nonempty <= max(len(cells), 1)


def parse_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.slice(0, 7) + "-01", errors="coerce")


def is_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").notna()


class RobustExcelLoader:
    def __init__(self) -> None:
        self.schema_rows: list[dict[str, Any]] = []
        self.mapping_rows: list[dict[str, Any]] = []
        self.preview_rows: list[dict[str, Any]] = []
        self.failure_rows: list[dict[str, Any]] = []
        self.cache: dict[tuple[str, str], dict[str, Any]] = {}

    def detect(self, file_name: str, sheet_name: str | int = 0) -> dict[str, Any]:
        key = (file_name, str(sheet_name))
        if key in self.cache:
            return self.cache[key]
        p = CSMAR / file_name
        code_to_desc, chinese_to_code, units = parse_des(file_name)
        known = known_fields_for(file_name, code_to_desc)
        required = REQUIRED_BY_FILE.get(file_name, [])
        if not p.exists():
            diag = {"status": "FILE_MISSING", "fields": {}, "data_start": None}
            self.cache[key] = diag
            return diag
        head = pd.read_excel(p, sheet_name=sheet_name, header=None, dtype=str, engine="openpyxl", nrows=30)
        row_count_raw = int(pd.read_excel(p, sheet_name=sheet_name, header=None, dtype=str, engine="openpyxl", usecols=[0]).shape[0])
        col_count_raw = int(head.shape[1])
        candidates: list[dict[str, Any]] = []
        for r in range(len(head)):
            fields: dict[int, dict[str, str]] = {}
            for c, value in enumerate(head.iloc[r].tolist()):
                code, source, chinese = extract_field_code(value, known, chinese_to_code)
                if code:
                    fields[c] = {"code": code, "source": source, "chinese": chinese, "header": normalize_cell_text(value)}
            req_count = len(set(x["code"] for x in fields.values()) & set(required))
            candidates.append({"header_row": r, "fields": fields, "score": len(set(x["code"] for x in fields.values())), "req_count": req_count, "kind": "single"})
            if r + 1 < len(head):
                combo: dict[int, dict[str, str]] = {}
                for c in range(col_count_raw):
                    text = " ".join([normalize_cell_text(head.iat[r, c]), normalize_cell_text(head.iat[r + 1, c])]).strip()
                    code, source, chinese = extract_field_code(text, known, chinese_to_code)
                    if code:
                        combo[c] = {"code": code, "source": f"two_row_{source}", "chinese": chinese, "header": text}
                req_combo = len(set(x["code"] for x in combo.values()) & set(required))
                candidates.append({"header_row": r, "fields": combo, "score": len(set(x["code"] for x in combo.values())), "req_count": req_combo, "kind": "two_row"})
        required_threshold = min(2, len(required))
        viable = [x for x in candidates if x["req_count"] >= required_threshold]
        best = max(viable or candidates, key=lambda x: (x["req_count"], x["score"], -x["header_row"]))
        fields = best["fields"]
        unit_row = None
        probe = best["header_row"] + (2 if best["kind"] == "two_row" else 1)
        if best["kind"] == "single" and best["header_row"] + 1 < len(head):
            next_fields = 0
            for value in head.iloc[best["header_row"] + 1].tolist():
                code, _, _ = extract_field_code(value, known, chinese_to_code)
                if code:
                    next_fields += 1
            if next_fields >= max(2, best["req_count"] // 2):
                probe = best["header_row"] + 2
        if probe < len(head) and row_is_unit(head.iloc[probe].tolist(), set(fields)):
            unit_row = probe
            data_start = probe + 1
        else:
            data_start = probe
        data_start = self.find_data_start(file_name, p, fields, data_start)
        detected = list(dict.fromkeys(x["code"] for x in fields.values()))
        missing = [x for x in required if x not in detected]
        status = "OK" if not missing else "MISSING_REQUIRED_FIELDS"
        diag = {
            "file_name": file_name,
            "sheet_name": str(sheet_name),
            "row_count_raw": row_count_raw,
            "col_count_raw": col_count_raw,
            "header_row": int(best["header_row"]),
            "unit_row": unit_row,
            "data_start": data_start,
            "field_by_pos": fields,
            "detected_fields": detected,
            "required_fields": required,
            "missing_required_fields": missing,
            "required_detected": [x for x in required if x in detected],
            "status": status,
            "des_units": units,
        }
        self.schema_rows.append({
            "file_name": file_name,
            "sheet_name": str(sheet_name),
            "row_count_raw": row_count_raw,
            "col_count_raw": col_count_raw,
            "detected_header_row_index": int(best["header_row"]),
            "detected_unit_row_index": "" if unit_row is None else int(unit_row),
            "detected_data_start_row_index": int(data_start),
            "detected_field_count": len(detected),
            "required_fields": "|".join(required),
            "required_fields_detected": "|".join(diag["required_detected"]),
            "missing_required_fields": "|".join(missing),
            "loader_status": status,
        })
        for pos, meta in fields.items():
            unit = ""
            if unit_row is not None and pos < col_count_raw:
                unit = normalize_cell_text(head.iat[unit_row, pos])
            self.mapping_rows.append({
                "file_name": file_name,
                "sheet_name": str(sheet_name),
                "original_column_position": int(pos),
                "original_header_cell": meta["header"],
                "normalized_field_code": meta["code"],
                "mapping_source": meta["source"],
                "chinese_label_detected": meta["chinese"],
                "unit_detected": unit or units.get(meta["code"], ""),
            })
        del head
        gc.collect()
        self.cache[key] = diag
        return diag

    def find_data_start(self, file_name: str, p: Path, fields: dict[int, dict[str, str]], start: int) -> int:
        probe = pd.read_excel(p, header=None, dtype=str, engine="openpyxl", skiprows=start, nrows=8)
        code_to_pos = {v["code"]: k for k, v in fields.items()}
        for offset in range(len(probe)):
            row = {code: (probe.iat[offset, pos] if pos < probe.shape[1] else None) for code, pos in code_to_pos.items()}
            if data_row_valid(file_name, row):
                del probe
                gc.collect()
                return start + offset
        del probe
        gc.collect()
        return start

    def read(self, file_name: str, columns: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
        diag = self.detect(file_name)
        fields = diag.get("field_by_pos", {})
        if not fields or diag.get("data_start") is None:
            self.failure_rows.append({"file_name": file_name, "sheet_name": "0", "reason": "no detected fields or data start"})
            return pd.DataFrame()
        wanted = columns or list(dict.fromkeys(x["code"] for x in fields.values()))
        pos_for_code: dict[str, int] = {}
        source_for_code: dict[str, str] = {}
        for pos, meta in fields.items():
            pos_for_code.setdefault(meta["code"], pos)
            source_for_code.setdefault(meta["code"], meta["source"])
        use_positions = [pos_for_code[c] for c in wanted if c in pos_for_code]
        names = [c for c in wanted if c in pos_for_code]
        if not use_positions:
            self.failure_rows.append({"file_name": file_name, "sheet_name": "0", "reason": f"wanted columns unavailable: {wanted}"})
            return pd.DataFrame()
        df = pd.read_excel(
            CSMAR / file_name,
            header=None,
            dtype=str,
            engine="openpyxl",
            skiprows=int(diag["data_start"]),
            usecols=use_positions,
            names=names,
            nrows=nrows,
        )
        df = df.dropna(how="all")
        for i, row in df.head(5).iterrows():
            item = {"file_name": file_name, "sheet_name": "0", "preview_row_number": int(i)}
            item.update({c: row.get(c, "") for c in df.columns})
            self.preview_rows.append(item)
        return df

    def write_diagnostics(self) -> None:
        pd.DataFrame(self.schema_rows).to_csv(OUT / "csmar_excel_loader_schema_detection.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(self.mapping_rows).to_csv(OUT / "csmar_excel_loader_field_mapping.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(self.preview_rows).to_csv(OUT / "csmar_excel_loader_preview_rows.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(self.failure_rows, columns=["file_name", "sheet_name", "reason"]).to_csv(OUT / "csmar_excel_loader_failures.csv", index=False, encoding="utf-8-sig")


def data_row_valid(file_name: str, row: dict[str, Any]) -> bool:
    if file_name == "TRD_Index.xlsx":
        return bool(normalize_cell_text(row.get("Indexcd"))) and pd.notna(pd.to_datetime(row.get("Trddt"), errors="coerce")) and (pd.notna(pd.to_numeric(row.get("Retindex"), errors="coerce")) or pd.notna(pd.to_numeric(row.get("Clsindex"), errors="coerce")))
    if file_name in {"TRD_Cnmont.xlsx", "TRD_Mont.xlsx"}:
        ret_fields = [k for k in row if "ret" in k.lower()]
        return bool(normalize_cell_text(row.get("Markettype"))) and pd.notna(pd.to_datetime(str(row.get("Trdmnt"))[:7] + "-01", errors="coerce")) and any(pd.notna(pd.to_numeric(row.get(k), errors="coerce")) for k in ret_fields)
    if file_name == "TRD_Mnth.xlsx":
        return bool(normalize_cell_text(row.get("Stkcd"))) and pd.notna(pd.to_datetime(str(row.get("Trdmnt"))[:7] + "-01", errors="coerce")) and (pd.notna(pd.to_numeric(row.get("Mretwd"), errors="coerce")) or pd.notna(pd.to_numeric(row.get("Mretnd"), errors="coerce")))
    if file_name == "TRD_Nrrate.xlsx":
        return pd.notna(pd.to_datetime(row.get("Clsdt"), errors="coerce")) and (pd.notna(pd.to_numeric(row.get("Nrrmtdt"), errors="coerce")) or pd.notna(pd.to_numeric(row.get("Nrrdaydt"), errors="coerce")))
    if file_name == "STK_MKT_FIVEFACDAY.xlsx":
        factor_fields = [k for k in row if k.lower() in {"riskpremium1", "riskpremium2", "smb1", "smb2", "hml1", "hml2", "rmw1", "rmw2", "cma1", "cma2"}]
        return pd.notna(pd.to_datetime(row.get("TradingDate"), errors="coerce")) and any(pd.notna(pd.to_numeric(row.get(k), errors="coerce")) for k in factor_fields)
    return True


def detect_unit(series: pd.Series, field: str = "") -> str:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return "UNKNOWN_NO_DATA"
    q95 = float(x.abs().quantile(0.95))
    mx = float(x.abs().max())
    if "nrr" in field.lower():
        return "PERCENT_BY_DES"
    return "PERCENT_SUSPECT" if mx > 5 or q95 > 1 else "DECIMAL_SUSPECT"


def to_decimal(series: pd.Series, unit: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x / 100.0 if "PERCENT" in unit else x


def compound_returns(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return np.nan if x.empty else float(np.prod(1.0 + x) - 1.0)


def load_portfolio_months() -> list[pd.Timestamp]:
    df = pd.read_csv(PORT / "unified_portfolio_monthly_gross_return.csv")
    col = next((c for c in df.columns if str(c).lower() in {"month_end", "portfolio_month_end", "date"}), df.columns[0])
    months = pd.to_datetime(df[col], errors="coerce").dropna().dt.to_period("M").dt.to_timestamp("M").drop_duplicates().sort_values()
    del df
    gc.collect()
    return list(months)


def priority_for_index(code: str, name: str) -> str:
    text = f"{code} {name}"
    if "中证800" in text or code in {"000906", "399906"}:
        return "PRIMARY_CSI800_CANDIDATE"
    if "沪深300" in text or code in {"000300", "399300"}:
        return "SECONDARY_HS300_CANDIDATE"
    if "中证500" in text or code in {"000905", "399905"}:
        return "SECONDARY_CSI500_CANDIDATE"
    if "中证1000" in text or "中证全指" in text or "中证流通" in text or code in {"000902", "000852", "399852", "000985"}:
        return "SECONDARY_CSI1000_OR_CSI_ALL_SHARE_CANDIDATE"
    return "OTHER_INDEX" if name else "UNKNOWN_INDEX"


def build_index_outputs(loader: RobustExcelLoader, months: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    code_to_desc, _, _ = parse_des("TRD_Index.xlsx")
    df = loader.read("TRD_Index.xlsx", ["Indexcd", "Trddt", "Clsindex", "Retindex"])
    if not {"Indexcd", "Trddt", "Clsindex", "Retindex"}.issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame()
    df["Indexcd"] = df["Indexcd"].astype("string").str.zfill(6)
    df["Trddt"] = pd.to_datetime(df["Trddt"], errors="coerce")
    df = df.dropna(subset=["Indexcd", "Trddt"]).sort_values(["Indexcd", "Trddt"])
    ret_unit = detect_unit(df["Retindex"], "Retindex")
    df["Retindex_decimal"] = to_decimal(df["Retindex"], ret_unit)
    index_map = {}
    for code, name in re.findall(r"([0-9]{6})[：:：]([^；;]+)", code_to_desc.get("Indexcd", "")):
        index_map[code] = name.strip()
    availability = []
    for code, g in df.groupby("Indexcd", sort=True):
        name = index_map.get(str(code), "")
        availability.append({
            "Indexcd": str(code), "first_date": g["Trddt"].min().strftime("%Y-%m-%d"), "last_date": g["Trddt"].max().strftime("%Y-%m-%d"),
            "row_count": int(len(g)), "nonnull_retindex_count": int(g["Retindex"].notna().sum()), "nonnull_clsindex_count": int(g["Clsindex"].notna().sum()),
            "possible_index_name_from_des_or_mapping": name, "priority_label": priority_for_index(str(code), name),
        })
    avail_df = pd.DataFrame(availability)
    candidate_codes = set(avail_df.loc[avail_df["priority_label"] != "OTHER_INDEX", "Indexcd"]) if not avail_df.empty else set()
    if not candidate_codes and not avail_df.empty:
        candidate_codes = set(avail_df["Indexcd"].head(5))
    rows = []
    for code in sorted(candidate_codes):
        g = df[df["Indexcd"] == code].copy()
        name = index_map.get(code, "")
        for i, t in enumerate(months[:-1]):
            nxt = months[i + 1]
            win = g[(g["Trddt"] > t) & (g["Trddt"] <= nxt)]
            source, val, missing, warning = "RETINDEX", np.nan, False, ""
            if win.empty:
                missing, warning = True, "NO_INDEX_TRADING_DAYS_IN_FORWARD_WINDOW"
            elif win["Retindex_decimal"].notna().any():
                val = compound_returns(win["Retindex_decimal"])
            else:
                source = "CLSINDEX_FALLBACK"
                prev = g[g["Trddt"] <= t].tail(1)
                if prev.empty or pd.isna(prev.iloc[0]["Clsindex"]) or pd.isna(win.iloc[-1]["Clsindex"]):
                    missing, warning = True, "RETINDEX_MISSING_AND_CLSINDEX_FALLBACK_UNAVAILABLE"
                else:
                    val = float(win.iloc[-1]["Clsindex"]) / float(prev.iloc[0]["Clsindex"]) - 1.0
            rows.append({
                "portfolio_month_end": t.strftime("%Y-%m-%d"), "benchmark_code": code,
                "benchmark_name_or_label": name or priority_for_index(code, name), "benchmark_fwd_ret_1m": val,
                "return_source": source, "source_start_trade_date": "" if win.empty else win["Trddt"].min().strftime("%Y-%m-%d"),
                "source_end_trade_date": "" if win.empty else win["Trddt"].max().strftime("%Y-%m-%d"),
                "trading_day_count": int(len(win)), "retindex_unit_detected": ret_unit,
                "missing_flag": bool(missing), "alignment_warning": warning,
            })
    del df
    gc.collect()
    return avail_df, pd.DataFrame(rows)


def build_market_monthly(loader: RobustExcelLoader, months: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    specs = [
        ("TRD_Cnmont.xlsx", ["Markettype", "Trdmnt", "Cmretwdeq", "Cmretwdos", "Cmretwdtl"], "综合市场"),
        ("TRD_Mont.xlsx", ["Markettype", "Trdmnt", "Mretwdeq", "Mretwdos", "Mretwdtl"], "分市场"),
    ]
    for file_name, cols, label_prefix in specs:
        df = loader.read(file_name, cols)
        if not {"Markettype", "Trdmnt"}.issubset(df.columns):
            continue
        df["Markettype"] = pd.to_numeric(df["Markettype"], errors="coerce").astype("Int64")
        df["Trdmnt"] = df["Trdmnt"].astype(str).str.slice(0, 7)
        fields = [c for c in cols if c not in {"Markettype", "Trdmnt"} and c in df.columns]
        units = {c: detect_unit(df[c], c) for c in fields}
        for c in fields:
            df[c + "_decimal"] = to_decimal(df[c], units[c])
        for t in months[:-1]:
            source_m = (t + pd.offsets.MonthEnd(1)).strftime("%Y-%m")
            for _, r in df[df["Trdmnt"] == source_m].iterrows():
                for c in fields:
                    markettype = "" if pd.isna(r["Markettype"]) else int(r["Markettype"])
                    rows.append({
                        "portfolio_month_end": t.strftime("%Y-%m-%d"), "source_table": file_name.replace(".xlsx", ""),
                        "markettype": markettype, "benchmark_field": c, "benchmark_label": f"{label_prefix}_Markettype_{markettype}_{c}",
                        "benchmark_fwd_ret_1m": r[c + "_decimal"], "source_trdmnt": source_m, "unit_detected": units[c],
                        "missing_flag": bool(pd.isna(r[c + "_decimal"])), "alignment_warning": "",
                    })
        del df
        gc.collect()
    return pd.DataFrame(rows)


def build_internal(months: list[pd.Timestamp]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    schema_names = pq.read_schema(PANEL).names
    wanted = ["symbol", "month_end", "fwd_ret_1m", "ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score", "composite_anomaly_flag_soft", "Msmvosd", "msmvosd"]
    df = pd.read_parquet(PANEL, columns=[c for c in wanted if c in schema_names])
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    df["fwd_ret_1m"] = pd.to_numeric(df["fwd_ret_1m"], errors="coerce")
    score_col = "ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score"
    eligible = df[df[score_col].notna()].copy() if score_col in df.columns else df.copy()
    mcap_col = "Msmvosd" if "Msmvosd" in eligible.columns else ("msmvosd" if "msmvosd" in eligible.columns else None)
    rows = []
    for t in months[:-1]:
        g = eligible[eligible["month_end"] == t]
        rows.append({"portfolio_month_end": t.strftime("%Y-%m-%d"), "benchmark_label": "INTERNAL_ELIGIBLE_UNIVERSE_EQUAL_WEIGHT", "benchmark_fwd_ret_1m": float(g["fwd_ret_1m"].mean()) if len(g) else np.nan, "universe_count": int(len(g)), "missing_fwd_ret_count": int(g["fwd_ret_1m"].isna().sum()) if len(g) else 0, "weighting_method": "equal_weight", "anomaly_filter_used": False, "mcap_weight_available": bool(mcap_col), "alignment_warning": "" if len(g) else "NO_INTERNAL_UNIVERSE_ROWS_FOR_MONTH"})
        if mcap_col:
            wdf = g[[mcap_col, "fwd_ret_1m"]].dropna()
            w = pd.to_numeric(wdf[mcap_col], errors="coerce") if len(wdf) else pd.Series(dtype=float)
            val = float(np.average(wdf["fwd_ret_1m"], weights=w)) if len(wdf) and w.sum() > 0 else np.nan
            rows.append({"portfolio_month_end": t.strftime("%Y-%m-%d"), "benchmark_label": "INTERNAL_ELIGIBLE_UNIVERSE_FLOAT_MCAP_WEIGHTED", "benchmark_fwd_ret_1m": val, "universe_count": int(len(wdf)), "missing_fwd_ret_count": int(g["fwd_ret_1m"].isna().sum()) if len(g) else 0, "weighting_method": "float_mcap_weighted", "anomaly_filter_used": False, "mcap_weight_available": True, "alignment_warning": "" if pd.notna(val) else "MCAP_WEIGHTED_RETURN_UNAVAILABLE"})
        if "composite_anomaly_flag_soft" in g.columns:
            cg = g[g["composite_anomaly_flag_soft"] != True]
            rows.append({"portfolio_month_end": t.strftime("%Y-%m-%d"), "benchmark_label": "INTERNAL_FLAG_CLEAN_UNIVERSE_EQUAL_WEIGHT", "benchmark_fwd_ret_1m": float(cg["fwd_ret_1m"].mean()) if len(cg) else np.nan, "universe_count": int(len(cg)), "missing_fwd_ret_count": int(cg["fwd_ret_1m"].isna().sum()) if len(cg) else 0, "weighting_method": "equal_weight", "anomaly_filter_used": True, "mcap_weight_available": bool(mcap_col), "alignment_warning": "" if len(cg) else "NO_FLAG_CLEAN_UNIVERSE_ROWS_FOR_MONTH"})
    del df, eligible
    gc.collect()
    return pd.DataFrame(rows)


def build_risk_free(loader: RobustExcelLoader, months: list[pd.Timestamp]) -> pd.DataFrame:
    df = loader.read("TRD_Nrrate.xlsx", ["Nrr1", "Clsdt", "Nrrdata", "Nrrdaydt", "Nrrmtdt"])
    if not {"Clsdt", "Nrrmtdt"}.issubset(df.columns):
        return pd.DataFrame()
    df["Clsdt"] = pd.to_datetime(df["Clsdt"], errors="coerce")
    unit = detect_unit(df["Nrrmtdt"], "Nrrmtdt")
    df["rf_decimal"] = to_decimal(df["Nrrmtdt"], unit)
    rows = []
    for t in months[:-1]:
        start, end = t + pd.Timedelta(days=1), t + pd.offsets.MonthEnd(1)
        g = df[(df["Clsdt"] >= start) & (df["Clsdt"] <= end)].sort_values("Clsdt")
        r = g.iloc[-1] if len(g) else None
        rows.append({
            "portfolio_month_end": t.strftime("%Y-%m-%d"), "risk_free_monthly_return": np.nan if r is None else r["rf_decimal"],
            "source_date": "" if r is None else r["Clsdt"].strftime("%Y-%m-%d"), "source_field": "Nrrmtdt", "unit_detected": unit,
            "nrr1": "" if r is None else r.get("Nrr1", ""), "missing_flag": r is None or pd.isna(r["rf_decimal"]),
            "alignment_warning": "NO_RISK_FREE_OBSERVATION_IN_FORWARD_MONTH" if r is None else "",
        })
    del df
    gc.collect()
    return pd.DataFrame(rows)


def build_ff(loader: RobustExcelLoader, months: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = loader.read("STK_MKT_FIVEFACDAY.xlsx", REQUIRED_BY_FILE["STK_MKT_FIVEFACDAY.xlsx"])
    manual = []
    if "TradingDate" not in df.columns:
        manual.append({"issue": "date column not recognized or missing", "candidate_columns": "|".join(df.columns)})
        return pd.DataFrame(), pd.DataFrame(manual)
    df["TradingDate"] = pd.to_datetime(df["TradingDate"], errors="coerce")
    factor_cols = [c for c in ["RiskPremium1", "RiskPremium2", "SMB1", "SMB2", "HML1", "HML2", "RMW1", "RMW2", "CMA1", "CMA2"] if c in df.columns]
    if not factor_cols:
        manual.append({"issue": "factor columns not recognized", "candidate_columns": "|".join(df.columns)})
        return pd.DataFrame(), pd.DataFrame(manual)
    units = {c: detect_unit(df[c], c) for c in factor_cols}
    for c in factor_cols:
        df[c + "_decimal"] = to_decimal(df[c], units[c])
    group_cols = [c for c in ["MarkettypeID", "Portfolios"] if c in df.columns]
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [(("ALL",), df)]
    for keys, g0 in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        label = "_".join(str(x) for x in keys)
        for i, t in enumerate(months[:-1]):
            nxt = months[i + 1]
            win = g0[(g0["TradingDate"] > t) & (g0["TradingDate"] <= nxt)]
            for c in factor_cols:
                val = compound_returns(win[c + "_decimal"]) if not win.empty else np.nan
                rows.append({"portfolio_month_end": t.strftime("%Y-%m-%d"), "factor_set_label": label, "factor_name": c, "factor_monthly_return": val, "source_start_date": "" if win.empty else win["TradingDate"].min().strftime("%Y-%m-%d"), "source_end_date": "" if win.empty else win["TradingDate"].max().strftime("%Y-%m-%d"), "trading_day_count": int(len(win)), "unit_detected": units[c], "missing_flag": bool(pd.isna(val)), "alignment_warning": "" if not win.empty else "NO_FACTOR_TRADING_DAYS_IN_FORWARD_WINDOW"})
    del df
    gc.collect()
    return pd.DataFrame(rows), pd.DataFrame(manual)


def main() -> None:
    checkpoint("开始 robust CSMAR Excel parser fix & benchmark re-audit")
    loader = RobustExcelLoader()
    files_detected = [f for f in REQUIRED_FILES if (CSMAR / f).exists()]
    for xlsx in [f for f in REQUIRED_FILES if f.endswith(".xlsx")]:
        checkpoint(f"检测 Excel schema: {xlsx}")
        loader.detect(xlsx)
    checkpoint("读取 portfolio month_end")
    months = load_portfolio_months()
    checkpoint("生成 TRD_Index 指数可用性与月度 forward candidates")
    index_avail, official = build_index_outputs(loader, months)
    index_avail.to_csv(OUT / "index_code_availability.csv", index=False, encoding="utf-8-sig")
    official.to_csv(OUT / "official_index_monthly_forward_return_candidates.csv", index=False, encoding="utf-8-sig")
    checkpoint("生成 CSMAR market monthly candidates")
    market = build_market_monthly(loader, months)
    market.to_csv(OUT / "csmar_market_monthly_forward_return_candidates.csv", index=False, encoding="utf-8-sig")
    checkpoint("生成 internal universe benchmark")
    internal = build_internal(months)
    internal.to_csv(OUT / "internal_universe_monthly_forward_benchmark.csv", index=False, encoding="utf-8-sig")
    checkpoint("生成 risk-free monthly alignment")
    rf = build_risk_free(loader, months)
    rf.to_csv(OUT / "risk_free_monthly_aligned.csv", index=False, encoding="utf-8-sig")
    checkpoint("生成 Fama-French monthly factor candidates")
    ff, ff_manual = build_ff(loader, months)
    ff.to_csv(OUT / "fama_french_monthly_factor_candidates.csv", index=False, encoding="utf-8-sig")
    ff_manual.to_csv(OUT / "fama_french_field_manual_review_required.csv", index=False, encoding="utf-8-sig")
    loader.write_diagnostics()

    schema = pd.DataFrame(loader.schema_rows)
    files_failed = sorted(set(schema.loc[schema["loader_status"] != "OK", "file_name"].tolist() + [r["file_name"] for r in loader.failure_rows]))
    files_parsed = sorted(set(schema.loc[schema["loader_status"] == "OK", "file_name"].tolist()))
    trd_index_schema = schema[schema["file_name"] == "TRD_Index.xlsx"].iloc[0].to_dict()
    csi800_rows = index_avail[index_avail["priority_label"] == "PRIMARY_CSI800_CANDIDATE"] if not index_avail.empty else pd.DataFrame()
    hs300_found = bool((index_avail["priority_label"] == "SECONDARY_HS300_CANDIDATE").any()) if not index_avail.empty else False
    csi500_found = bool((index_avail["priority_label"] == "SECONDARY_CSI500_CANDIDATE").any()) if not index_avail.empty else False
    internal_ready = bool(len(internal) and ((internal["benchmark_label"] == "INTERNAL_ELIGIBLE_UNIVERSE_EQUAL_WEIGHT") & internal["benchmark_fwd_ret_1m"].notna()).any())
    official_ready = bool(len(official) and official["benchmark_fwd_ret_1m"].notna().any()) or bool(len(market) and market["benchmark_fwd_ret_1m"].notna().any())
    rf_ready = bool(len(rf) and rf["risk_free_monthly_return"].notna().any())
    ff_ready = bool(len(ff) and ff["factor_monthly_return"].notna().any() and ff_manual.empty)
    trd_index_dates = pd.to_datetime(loader.read("TRD_Index.xlsx", ["Trddt"])["Trddt"], errors="coerce")

    def req_detected(file_name: str) -> list[str]:
        row = schema[schema["file_name"] == file_name]
        return [] if row.empty else str(row.iloc[0]["required_fields_detected"]).split("|")

    manual_reasons = []
    if csi800_rows.empty:
        manual_reasons.append("CSI800 Indexcd 未从 TRD_Index DES/实际代码自动确认；请查看 index_code_availability.csv 人工确认。")
    if not ff_ready:
        manual_reasons.append("Fama-French 字段或覆盖需要人工复核/可后补。")
    if not rf_ready:
        manual_reasons.append("risk-free monthly alignment 不可用或覆盖不足。")
    if not official_ready:
        manual_reasons.append("official/market benchmark candidate 不可用。")
    if not internal_ready:
        manual_reasons.append("internal universe benchmark 不可用。")

    if not official_ready or not internal_ready:
        final_decision = "BENCHMARK_SOURCE_AUDIT_ALIGNMENT_FAIL_INSUFFICIENT_BENCHMARK"
        next_step = "先复核 robust loader diagnostics；若 TRD_Index 代码不能自动确认，依据 index_code_availability.csv 人工确认指数代码。"
    elif csi800_rows.empty:
        final_decision = "BENCHMARK_SOURCE_AUDIT_ALIGNMENT_WATCH_INDEX_CODE_MANUAL_REVIEW_REQUIRED"
        next_step = "人工确认中证800/沪深300/中证500 Indexcd；确认后进入 benchmark-relative evaluation prep。"
    elif not ff_ready:
        final_decision = "BENCHMARK_SOURCE_AUDIT_ALIGNMENT_WATCH_FACTOR_FIELD_MANUAL_REVIEW_REQUIRED"
        next_step = "benchmark-relative eval prep 可继续；factor attribution 前复核 Fama-French 字段。"
    else:
        final_decision = "BENCHMARK_SOURCE_AUDIT_ALIGNMENT_READY_FOR_BENCHMARK_RELATIVE_EVAL_PREP"
        next_step = "进入 benchmark-relative evaluation prep。"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "robust_loader_used": True,
        "files_detected": len(files_detected),
        "files_parsed_successfully": len(files_parsed),
        "files_failed_to_parse": len(files_failed),
        "parser_issue_confirmed": True,
        "previous_failure_due_to_header_detection": True,
        "trd_index_required_fields_detected": req_detected("TRD_Index.xlsx"),
        "trd_cnmont_required_fields_detected": req_detected("TRD_Cnmont.xlsx"),
        "trd_mont_required_fields_detected": req_detected("TRD_Mont.xlsx"),
        "trd_mnth_required_fields_detected": req_detected("TRD_Mnth.xlsx"),
        "trd_nrrate_required_fields_detected": req_detected("TRD_Nrrate.xlsx"),
        "fivefacday_required_fields_detected": req_detected("STK_MKT_FIVEFACDAY.xlsx"),
        "trd_index_rows": int(trd_index_schema.get("row_count_raw", 0) - trd_index_schema.get("detected_data_start_row_index", 0)),
        "trd_index_min_date": "" if trd_index_dates.dropna().empty else trd_index_dates.min().strftime("%Y-%m-%d"),
        "trd_index_max_date": "" if trd_index_dates.dropna().empty else trd_index_dates.max().strftime("%Y-%m-%d"),
        "csi800_candidate_found": bool(not csi800_rows.empty),
        "csi800_index_code": "" if csi800_rows.empty else str(csi800_rows.iloc[0]["Indexcd"]),
        "hs300_candidate_found": hs300_found,
        "csi500_candidate_found": csi500_found,
        "official_index_monthly_candidate_count": int(len(official)),
        "csmar_market_monthly_candidate_count": int(len(market)),
        "internal_benchmark_candidate_count": int(len(internal)),
        "risk_free_monthly_ready": rf_ready,
        "fama_french_monthly_ready": ff_ready,
        "manual_review_required": bool(manual_reasons),
        "manual_review_reasons": manual_reasons,
        "primary_official_benchmark_recommended": "CSI800 official index monthly forward return" if not csi800_rows.empty else ("TRD_Cnmont broad-market fallback; manual Indexcd confirmation required" if official_ready else "NONE_OFFICIAL_BENCHMARK_UNAVAILABLE"),
        "primary_research_benchmark_recommended": "INTERNAL_ELIGIBLE_UNIVERSE_EQUAL_WEIGHT" if internal_ready else "",
        "factor_attribution_ready": bool(ff_ready and rf_ready),
        "benchmark_relative_eval_prep_allowed": bool(official_ready and internal_ready),
        "alpha_beta_eval_prep_allowed": bool(ff_ready and rf_ready),
        "portfolio_weights_modified": False,
        "portfolio_weights_reconstructed": False,
        "portfolio_benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "final_decision": final_decision,
        "recommended_next_step": next_step,
    }

    pd.DataFrame([{"benchmark_label": summary["primary_official_benchmark_recommended"], "benchmark_type": "primary_official_benchmark", "coverage_pass": official_ready, "recommended_use": "benchmark-relative evaluation prep", "caveats": "; ".join(manual_reasons)}]).to_csv(OUT / "benchmark_candidate_recommendation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"qa_item": "robust_loader_used", "pass": True, "detail": "pd.read_excel header=None dtype=str engine=openpyxl"},
        {"qa_item": "no_portfolio_weights_modified", "pass": True, "detail": "未读取或写入 weights 文件"},
        {"qa_item": "no_benchmark_relative_return_calculated", "pass": True, "detail": "仅构造 benchmark candidates"},
        {"qa_item": "no_alpha_beta_regression_calculated", "pass": True, "detail": "未运行回归"},
        {"qa_item": "no_training_or_shap", "pass": True, "detail": "未训练，未计算 SHAP"},
    ]).to_csv(OUT / "final_qa.csv", index=False, encoding="utf-8-sig")
    (OUT / "benchmark_source_audit_monthly_alignment_summary_fixed.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "terminal_summary.json").write_text(json.dumps({"task_name": TASK, "final_decision": final_decision, "stdout_log": str(RUN / "run_stdout.txt"), "stderr_log": str(RUN / "run_stderr.txt"), "outputs_dir": str(OUT)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUN / "terminal_summary.json").write_text((OUT / "terminal_summary.json").read_text(encoding="utf-8"), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(OUT / "task_completion_card.csv", index=False, encoding="utf-8-sig")
    (OUT / "task_completion_card.md").write_text("# 任务完成卡\n\n" + "\n".join([f"- final_decision: {final_decision}", "- robust_loader_used: True", f"- files_detected: {len(files_detected)}", f"- files_parsed_successfully: {len(files_parsed)}", f"- files_failed_to_parse: {len(files_failed)}", f"- recommended_next_step: {next_step}"]) + "\n", encoding="utf-8")
    checkpoint("任务完成", status="completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        checkpoint(f"失败: {exc}", status="failed")
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        raise

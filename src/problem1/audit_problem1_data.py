#!/usr/bin/env python3
"""Audit and independently rebuild the Question 1 sales-data tables.

The script intentionally preserves every source record.  It creates diagnostic
outputs and reconstructed tables in a separate directory; it never overwrites
the team-maintained files in ``问题1_建模数据``.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


SALES = "销售"
RETURNS = "退货"
YES = "是"


def read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, dtype={"单品编码": "string", "分类编码": "string"})


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def fmt_number(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{digits}f}"
    return str(value)


def time_parse_status(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = series.astype("string").str.strip()
    parsed = pd.to_datetime(raw, format="%H:%M:%S.%f", errors="coerce")
    unresolved = parsed.isna() & raw.notna()
    parsed.loc[unresolved] = pd.to_datetime(raw.loc[unresolved], format="%H:%M:%S", errors="coerce")
    return parsed, raw.notna() & parsed.isna()


def source_audit(products: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    product_keys = products["单品编码"].astype("string")
    code_to_name = products.groupby("单品编码")["单品名称"].nunique(dropna=False)
    code_to_cat = products.groupby("单品编码")[["分类编码", "分类名称"]].nunique(dropna=False)
    cat_code_to_name = products.groupby("分类编码")["分类名称"].nunique(dropna=False)
    names_to_codes = products.groupby("单品名称")["单品编码"].nunique(dropna=False)
    tx_codes = set(transactions["单品编码"].astype("string").dropna())
    product_code_set = set(product_keys.dropna())
    bad_code = product_keys.str.fullmatch(r"\d{15}", na=False).eq(False)
    return {
        "rows": len(products),
        "distinct_product_codes": product_keys.nunique(dropna=True),
        "duplicate_full_rows": int(products.duplicated().sum()),
        "missing_by_column": products.isna().sum().to_dict(),
        "product_codes_with_multiple_names": int((code_to_name > 1).sum()),
        "product_codes_with_multiple_category_codes": int((code_to_cat["分类编码"] > 1).sum()),
        "product_codes_with_multiple_category_names": int((code_to_cat["分类名称"] > 1).sum()),
        "category_codes_with_multiple_names": int((cat_code_to_name > 1).sum()),
        "product_names_with_multiple_codes": int((names_to_codes > 1).sum()),
        "non_15_digit_product_code_rows": int(bad_code.sum()),
        "transaction_distinct_product_codes": len(tx_codes),
        "transaction_codes_missing_from_attachment1": sorted(tx_codes - product_code_set),
        "attachment1_codes_absent_from_attachment2": sorted(product_code_set - tx_codes),
    }


def audit_transactions(transactions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    tx = transactions.copy()
    tx["日期"] = pd.to_datetime(tx["销售日期"], errors="coerce").dt.normalize()
    tx["单品编码"] = tx["单品编码"].astype("string")
    tx["销量(千克)"] = pd.to_numeric(tx["销量(千克)"], errors="coerce")
    tx["销售单价(元/千克)"] = pd.to_numeric(tx["销售单价(元/千克)"], errors="coerce")
    parsed_time, invalid_time = time_parse_status(tx["扫码销售时间"])
    tx["parsed_time"] = parsed_time
    full_dates = pd.date_range(tx["日期"].min(), tx["日期"].max(), freq="D")
    observed_dates = pd.DatetimeIndex(tx["日期"].dropna().unique())
    no_records_dates = full_dates.difference(observed_dates)
    q = tx["销量(千克)"]
    p = tx["销售单价(元/千克)"]
    types = tx["销售类型"].astype("string")
    discounts = tx["是否打折销售"].astype("string")
    sales_q = q[types.eq(SALES)]
    return_q = q[types.eq(RETURNS)]
    # Pricing variation relevant to the demand tables is assessed on normal sales.
    unit_price_by_item_day = tx.loc[types.eq(SALES)].groupby(["日期", "单品编码"])["销售单价(元/千克)"].nunique(dropna=True)
    # Tukey fences are diagnostics, never filters.
    finite_q = q.dropna()
    q1, q3 = finite_q.quantile([0.25, 0.75])
    upper_fence = q3 + 3 * (q3 - q1)
    finite_p = p.dropna()
    p1, p3 = finite_p.quantile([0.25, 0.75])
    price_upper_fence = p3 + 3 * (p3 - p1)
    audit = {
        "rows": len(tx),
        "date_min": str(tx["日期"].min().date()),
        "date_max": str(tx["日期"].max().date()),
        "calendar_days_in_range": len(full_dates),
        "dates_with_any_raw_record": len(observed_dates),
        "dates_with_no_raw_record": [str(x.date()) for x in no_records_dates],
        "invalid_sales_date_rows": int(tx["日期"].isna().sum()),
        "invalid_scan_time_rows": int(invalid_time.sum()),
        "sales_type_counts": types.value_counts(dropna=False).to_dict(),
        "discount_value_counts": discounts.value_counts(dropna=False).to_dict(),
        "discount_record_share": float(discounts.eq(YES).mean()),
        "sales_negative_quantity_rows": int((sales_q < 0).sum()),
        "return_negative_quantity_rows": int((return_q < 0).sum()),
        "return_nonnegative_quantity_rows": int((return_q >= 0).sum()),
        "quantity_summary": q.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_dict(),
        "price_summary": p.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_dict(),
        "quantity_gt_tukey_3iqr_rows": int((q > upper_fence).sum()),
        "quantity_tukey_3iqr_upper_fence": float(upper_fence),
        "price_le_zero_rows": int((p <= 0).sum()),
        "price_gt_tukey_3iqr_rows": int((p > price_upper_fence).sum()),
        "price_tukey_3iqr_upper_fence": float(price_upper_fence),
        "item_days_with_multiple_prices": int((unit_price_by_item_day > 1).sum()),
        "item_days_total": len(unit_price_by_item_day),
    }
    return audit, tx


def exception_samples(tx: pd.DataFrame, audit: dict, limit: int = 20) -> pd.DataFrame:
    """Return traceable samples for review without changing source values."""
    keep = ["销售日期", "扫码销售时间", "单品编码", "销量(千克)", "销售单价(元/千克)", "销售类型", "是否打折销售"]
    groups = [
        ("退货记录", tx["销售类型"].astype("string").eq(RETURNS)),
        ("正常销售负销量", tx["销售类型"].astype("string").eq(SALES) & tx["销量(千克)"].lt(0)),
        ("数量高于3IQR上界", tx["销量(千克)"].gt(audit["quantity_tukey_3iqr_upper_fence"])),
        ("价格高于3IQR上界", tx["销售单价(元/千克)"].gt(audit["price_tukey_3iqr_upper_fence"])),
        ("非正价格", tx["销售单价(元/千克)"].le(0)),
    ]
    parts = []
    for label, mask in groups:
        sample = tx.loc[mask, keep].head(limit).copy()
        sample.insert(0, "审计标记", label)
        parts.append(sample)
    return pd.concat(parts, ignore_index=True)


def build_daily_tables(tx: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = tx.loc[tx["销售类型"].astype("string").eq(SALES)].copy()
    sales["销售额"] = sales["销量(千克)"] * sales["销售单价(元/千克)"]
    sales["是否打折"] = sales["是否打折销售"].astype("string").eq(YES)
    sales["打折交易笔数"] = sales["是否打折"].astype(int)
    product_cols = ["单品编码", "单品名称", "分类编码", "分类名称"]
    product_map = products[product_cols].copy()
    product_map["单品编码"] = product_map["单品编码"].astype("string")
    sales = sales.merge(product_map, on="单品编码", how="left", validate="many_to_one", indicator=True)
    unmatched = sales.loc[sales["_merge"].ne("both")].copy()
    if not unmatched.empty:
        raise ValueError("Normal sales contain product codes absent from Attachment 1.")
    keys = ["日期", "单品编码", "单品名称", "分类编码", "分类名称"]
    item = sales.groupby(keys, as_index=False).agg(
        当日正常销售量=("销量(千克)", "sum"),
        当日销售额=("销售额", "sum"),
        当日交易笔数=("销量(千克)", "size"),
        当日打折交易笔数=("打折交易笔数", "sum"),
        当日打折销量=("销量(千克)", lambda s: s[sales.loc[s.index, "是否打折"]].sum()),
    )
    item["当日打折销量占比"] = item["当日打折销量"] / item["当日正常销售量"]
    item["当日平均零售单价"] = item["当日销售额"] / item["当日正常销售量"]
    item = item.sort_values(["日期", "单品编码"]).reset_index(drop=True)
    cat = item.groupby(["日期", "分类编码", "分类名称"], as_index=False).agg(
        当日总销量=("当日正常销售量", "sum"),
        当日销售额=("当日销售额", "sum"),
        当日交易笔数=("当日交易笔数", "sum"),
        当日打折交易笔数=("当日打折交易笔数", "sum"),
        在售单品数=("单品编码", "nunique"),
        当日打折销量=("当日打折销量", "sum"),
    )
    cat["当日打折销量占比"] = cat["当日打折销量"] / cat["当日总销量"]
    cat["当日加权平均零售单价"] = cat["当日销售额"] / cat["当日总销量"]
    cat = cat.sort_values(["日期", "分类编码"]).reset_index(drop=True)
    returns = tx.loc[tx["销售类型"].astype("string").eq(RETURNS)].copy()
    returns["退货量"] = returns["销量(千克)"]
    daily_reconciliation = sales.groupby("日期", as_index=False).agg(正常销售量=("销量(千克)", "sum"))
    ret_daily = returns.groupby("日期", as_index=False).agg(退货量=("退货量", "sum"))
    daily_reconciliation = daily_reconciliation.merge(ret_daily, on="日期", how="outer").fillna(0)
    daily_reconciliation["净销量"] = daily_reconciliation["正常销售量"] + daily_reconciliation["退货量"]
    return item, cat, daily_reconciliation, sales


def build_calendar_and_activity(tx: pd.DataFrame, item: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start, end = tx["日期"].min(), tx["日期"].max()
    dates = pd.DataFrame({"日期": pd.date_range(start, end, freq="D")})
    raw_dates = set(tx["日期"].dropna())
    sales_dates = set(tx.loc[tx["销售类型"].astype("string").eq(SALES), "日期"].dropna())
    calendar = dates.copy()
    calendar["是否有原始流水记录"] = calendar["日期"].isin(raw_dates)
    calendar["是否有销售记录"] = calendar["日期"].isin(sales_dates)
    categories = products[["分类编码", "分类名称"]].drop_duplicates().sort_values("分类编码")
    full_panel = dates.assign(_key=1).merge(categories.assign(_key=1), on="_key").drop(columns="_key")
    full_panel = full_panel.merge(item.groupby(["日期", "分类编码", "分类名称"], as_index=False)["当日正常销售量"].sum(), on=["日期", "分类编码", "分类名称"], how="left")
    # A global no-sale-record date is not evidence of zero demand.
    full_panel.loc[~full_panel["日期"].isin(sales_dates), "当日正常销售量"] = np.nan
    full_panel.loc[full_panel["日期"].isin(sales_dates) & full_panel["当日正常销售量"].isna(), "当日正常销售量"] = 0.0
    active = item.groupby(["单品编码", "单品名称", "分类编码", "分类名称"], as_index=False).agg(
        first_sale_date=("日期", "min"),
        last_sale_date=("日期", "max"),
        sales_days=("日期", "nunique"),
    )
    active["active_span_calendar_days"] = (active["last_sale_date"] - active["first_sale_date"]).dt.days + 1
    normal_sales_dates = pd.DatetimeIndex(sorted(sales_dates))
    active["active_span_observed_sales_days"] = active.apply(lambda r: int(((normal_sales_dates >= r["first_sale_date"]) & (normal_sales_dates <= r["last_sale_date"])).sum()), axis=1)
    active["sales_days_share_of_active_span"] = active["sales_days"] / active["active_span_calendar_days"]
    active = active.sort_values(["sales_days", "单品编码"], ascending=[False, True]).reset_index(drop=True)
    sales_dates_by_item = item.groupby("单品编码")["日期"].agg(lambda x: set(x)).to_dict()
    active_index = active.set_index("单品编码")
    pair_rows = []
    codes = active["单品编码"].tolist()
    for left, right in combinations(codes, 2):
        a, b = active_index.loc[left], active_index.loc[right]
        overlap_start, overlap_end = max(a.first_sale_date, b.first_sale_date), min(a.last_sale_date, b.last_sale_date)
        calendar_overlap = max(0, (overlap_end - overlap_start).days + 1)
        observed_overlap = int(((normal_sales_dates >= overlap_start) & (normal_sales_dates <= overlap_end)).sum()) if calendar_overlap else 0
        pair_rows.append({
            "单品编码_A": left, "单品编码_B": right,
            "共同有销售记录的日期数": len(sales_dates_by_item[left] & sales_dates_by_item[right]),
            "共同活跃期日历天数": calendar_overlap,
            "共同活跃期有销售记录天数": observed_overlap,
        })
    return calendar, full_panel, active, pd.DataFrame(pair_rows)


def summary_tables(item: pd.DataFrame, cat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def summaries(frame: pd.DataFrame, keys: list[str], qty: str, amount: str) -> pd.DataFrame:
        output = frame.groupby(keys, as_index=False).agg(
            有销售日数=("日期", "nunique"), 总销量=(qty, "sum"), 总销售额=(amount, "sum"),
            有销量日均销量=(qty, "mean"), 单日最大销量=(qty, "max"), 日销量标准差=(qty, "std"),
        )
        output["日销量变异系数"] = output["日销量标准差"] / output["有销量日均销量"]
        return output
    return summaries(item, ["单品编码", "单品名称", "分类编码", "分类名称"], "当日正常销售量", "当日销售额"), summaries(cat, ["分类编码", "分类名称"], "当日总销量", "当日销售额")


def compare_existing(rebuilt: pd.DataFrame, existing_path: Path, key_cols: list[str], qty_col: str, amount_col: str) -> dict:
    if not existing_path.exists():
        return {"status": "not_run", "reason": f"Existing file not present: {existing_path}"}
    existing = pd.read_csv(existing_path, encoding="utf-8-sig", dtype={"单品编码": "string", "分类编码": "string"})
    # Existing files use concise field names; only control totals and unique keys are required for an auditable first pass.
    expected_qty = "销量" if "销量" in existing.columns else qty_col
    expected_amount = "销售额" if "销售额" in existing.columns else amount_col
    result = {
        "status": "run", "existing_rows": len(existing), "rebuilt_rows": len(rebuilt),
        "existing_duplicate_keys": int(existing.duplicated(key_cols).sum()),
        "rebuilt_duplicate_keys": int(rebuilt.duplicated(key_cols).sum()),
        "existing_distinct_entities": int(existing[key_cols[-1]].nunique()) if key_cols[-1] in existing else None,
        "rebuilt_distinct_entities": int(rebuilt[key_cols[-1]].nunique()),
    }
    if expected_qty in existing:
        result["existing_total_quantity"] = float(pd.to_numeric(existing[expected_qty], errors="coerce").sum())
        result["rebuilt_total_quantity"] = float(rebuilt[qty_col].sum())
        result["quantity_difference"] = result["existing_total_quantity"] - result["rebuilt_total_quantity"]
    if expected_amount in existing:
        result["existing_total_amount"] = float(pd.to_numeric(existing[expected_amount], errors="coerce").sum())
        result["rebuilt_total_amount"] = float(rebuilt[amount_col].sum())
        result["amount_difference"] = result["existing_total_amount"] - result["rebuilt_total_amount"]
    name_map = {
        "当日正常销售量": "销量", "当日总销量": "销量", "当日销售额": "销售额",
        "当日交易笔数": "交易笔数", "当日打折交易笔数": "打折笔数",
        "当日打折销量": "打折销量", "当日平均零售单价": "平均零售单价(元/千克)",
        "当日加权平均零售单价": "平均零售单价(元/千克)", "当日打折销量占比": "打折销量占比",
        "在售单品数": "在售单品数",
    }
    merged = existing.merge(rebuilt, on=key_cols, how="outer", indicator=True, suffixes=("_existing", "_rebuilt"))
    result["unmatched_existing_keys"] = int(merged["_merge"].eq("left_only").sum())
    result["unmatched_rebuilt_keys"] = int(merged["_merge"].eq("right_only").sum())
    field_checks = {}
    for built_name, existing_name in name_map.items():
        if built_name not in rebuilt.columns:
            continue
        if existing_name not in existing.columns:
            field_checks[built_name] = {"status": "missing_from_existing"}
            continue
        left = pd.to_numeric(merged.get(f"{existing_name}_existing", merged.get(existing_name)), errors="coerce")
        right = pd.to_numeric(merged.get(f"{built_name}_rebuilt", merged.get(built_name)), errors="coerce")
        comparable = left.notna() & right.notna()
        delta = (left[comparable] - right[comparable]).abs()
        field_checks[built_name] = {
            "status": "checked", "comparable_rows": int(comparable.sum()),
            "max_abs_difference": float(delta.max()) if len(delta) else None,
            "rows_diff_over_1e-9": int((delta > 1e-9).sum()),
        }
    result["field_checks"] = field_checks
    return result


def markdown_report(path: Path, product_audit: dict, transaction_audit: dict, item_compare: dict, cat_compare: dict, item: pd.DataFrame, cat: pd.DataFrame, calendar: pd.DataFrame, active: pd.DataFrame, pairs: pd.DataFrame) -> None:
    missing_tx = product_audit["transaction_codes_missing_from_attachment1"]
    no_records = transaction_audit["dates_with_no_raw_record"]
    lines = [
        "# 问题一数据审计", "", "## 范围与原则", "",
        "本审计只使用附件1和附件2。正常销售记录用于问题一主分析；退货保留在独立对账表中。脚本不删除任何异常记录，也不覆盖既有建模文件。", "",
        "## 附件1审计", "",
        f"- 原始行数：{product_audit['rows']}；单品编码数：{product_audit['distinct_product_codes']}；完整重复行：{product_audit['duplicate_full_rows']}。",
        f"- 单品编码对应多个名称：{product_audit['product_codes_with_multiple_names']}；单品编码对应多个分类编码：{product_audit['product_codes_with_multiple_category_codes']}；分类编码对应多个名称：{product_audit['category_codes_with_multiple_names']}。",
        f"- 空值计数：`{json.dumps(product_audit['missing_by_column'], ensure_ascii=False)}`。非15位数字编码行：{product_audit['non_15_digit_product_code_rows']}。",
        f"- 附件2出现但附件1缺失的单品编码数：{len(missing_tx)}；编码：{missing_tx[:20]}。附件1未在附件2出现的编码数：{len(product_audit['attachment1_codes_absent_from_attachment2'])}。", "",
        "## 附件2原始流水审计", "",
        f"- 记录数：{transaction_audit['rows']}；日期范围：{transaction_audit['date_min']} 至 {transaction_audit['date_max']}；范围内日历日：{transaction_audit['calendar_days_in_range']}。",
        f"- 无任何原始流水日期：{len(no_records)} 天（{', '.join(no_records)}）。这些日期不能解释为“停业”，只能解释为“没有观测到流水记录”。",
        f"- 无法按 `HH:MM:SS[.ffffff]` 解析的扫码时间：{transaction_audit['invalid_scan_time_rows']} 行；销售类型：`{json.dumps(transaction_audit['sales_type_counts'], ensure_ascii=False)}`。",
        f"- 正常销售中的负销量：{transaction_audit['sales_negative_quantity_rows']} 行；退货中的负销量：{transaction_audit['return_negative_quantity_rows']} 行；退货中非负销量：{transaction_audit['return_nonnegative_quantity_rows']} 行。",
        f"- 小于等于零的销售单价：{transaction_audit['price_le_zero_rows']} 行；数量高于3×IQR上界 {fmt_number(transaction_audit['quantity_tukey_3iqr_upper_fence'])} 的记录：{transaction_audit['quantity_gt_tukey_3iqr_rows']} 行；单价高于3×IQR上界 {fmt_number(transaction_audit['price_tukey_3iqr_upper_fence'])} 的记录：{transaction_audit['price_gt_tukey_3iqr_rows']} 行。它们均已保留，只作为待解释标记。",
        f"- 同一单品同一天存在多个销售单价：{transaction_audit['item_days_with_multiple_prices']} / {transaction_audit['item_days_total']} 个单品日，因此日均价必须用销售额除以销量。",
        f"- 打折取值：`{json.dumps(transaction_audit['discount_value_counts'], ensure_ascii=False)}`；打折记录占比：{transaction_audit['discount_record_share']:.2%}。", "",
        "## 退货口径", "",
        "问题一的需求分布使用正常销售量。`每日正常销售量_退货量_净销量.csv` 同时保留正常销售量、带符号的退货量和净销量，因此退货没有从审计链条中消失。现有问题一说明写明其日汇总仅保留销售、剔除退货；此口径与主分析一致，但原表无法回答退货规模，需引用本审计输出。", "",
        "## 日表与总体表复算", "",
        f"- 重算单品×日：{len(item)} 行、{item['单品编码'].nunique()} 个单品，正常销售总量 {fmt_number(item['当日正常销售量'].sum())} 千克、销售额 {fmt_number(item['当日销售额'].sum())} 元，重复日期×单品：{int(item.duplicated(['日期','单品编码']).sum())}。",
        f"- 重算品类×日：{len(cat)} 行、{cat['分类编码'].nunique()} 个品类，重复日期×品类：{int(cat.duplicated(['日期','分类编码']).sum())}。",
        f"- 现有单品日表比较：`{json.dumps(item_compare, ensure_ascii=False)}`。",
        f"- 现有品类日表比较：`{json.dumps(cat_compare, ensure_ascii=False)}`。", "",
        "## 完整日期面板与单品活跃期", "",
        f"- 全日期范围 {len(calendar)} 天，其中有正常销售记录 {int(calendar['是否有销售记录'].sum())} 天。完整品类面板在全店无销售记录日保留 `NaN`；仅在全店有销售记录而某品类缺失时写为 0。字段应命名为“是否有销售记录”，不应断言为“是否营业”。",
        f"- 单品活跃表覆盖 {len(active)} 个实际销售单品；销售天数中位数为 {active['sales_days'].median():.0f} 天，活跃跨度内销售日占比中位数为 {active['sales_days_share_of_active_span'].median():.2%}。不应把 1095×单品全缺失一律填0后直接计算相关系数。",
        f"- 已输出 {len(pairs)} 个单品对的 overlap 统计，包含共同销售日、共同活跃期日历日和共同活跃期有销售记录日；尚未计算完整 Pearson 相关矩阵。", "",
        "## 可用性与建议", "",
        "现有表的口径说明与本审计的主分析口径一致：使用正常销售、金额/销量构造加权均价、全店无记录日不作为零销量。若脚本在含有既有CSV的分支路径运行，应以自动比较结果判定逐字段是否一致。优先使用重算输出和现有表的比较结果后，再开始问题一 EDA。建议将 `营业日历.csv` 的 `是否营业_有销售` 改名为 `是否有销售记录`，并在后续单品相关性分析设置共同活跃期/共同观测日阈值。", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing 附件1.xlsx and 附件2.xlsx")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/problem1_audit_outputs"))
    parser.add_argument("--report", type=Path, default=Path("analysis/problem1_data_audit.md"))
    parser.add_argument("--existing-dir", type=Path, default=Path("问题1_建模数据"), help="Optional current Question-1 CSV directory")
    args = parser.parse_args()
    products = read_excel(args.source_dir / "附件1.xlsx")
    transactions = read_excel(args.source_dir / "附件2.xlsx")
    product_audit = source_audit(products, transactions)
    transaction_audit, prepared_tx = audit_transactions(transactions)
    samples = exception_samples(prepared_tx, transaction_audit)
    item, cat, reconciliation, sales = build_daily_tables(prepared_tx, products)
    calendar, panel, active, pairs = build_calendar_and_activity(prepared_tx, item, products)
    item_stats, cat_stats = summary_tables(item, cat)
    out = args.output_dir
    write_csv(pd.DataFrame([product_audit | {"transaction_codes_missing_from_attachment1": ";".join(product_audit["transaction_codes_missing_from_attachment1"]), "attachment1_codes_absent_from_attachment2": ";".join(product_audit["attachment1_codes_absent_from_attachment2"]), "missing_by_column": json.dumps(product_audit["missing_by_column"], ensure_ascii=False)}]), out / "附件1_审计摘要.csv")
    tx_flat = transaction_audit.copy(); tx_flat["dates_with_no_raw_record"] = ";".join(tx_flat["dates_with_no_raw_record"]); tx_flat["sales_type_counts"] = json.dumps(tx_flat["sales_type_counts"], ensure_ascii=False); tx_flat["discount_value_counts"] = json.dumps(tx_flat["discount_value_counts"], ensure_ascii=False); tx_flat["quantity_summary"] = json.dumps(tx_flat["quantity_summary"], ensure_ascii=False); tx_flat["price_summary"] = json.dumps(tx_flat["price_summary"], ensure_ascii=False)
    write_csv(pd.DataFrame([tx_flat]), out / "附件2_审计摘要.csv")
    write_csv(samples, out / "附件2_异常记录样例.csv")
    write_csv(item, out / "单品_日销售_重算.csv")
    write_csv(cat, out / "品类_日销售_重算.csv")
    write_csv(reconciliation, out / "每日正常销售量_退货量_净销量.csv")
    write_csv(calendar, out / "日期记录覆盖审计.csv")
    write_csv(panel, out / "品类_完整日期面板_重算.csv")
    write_csv(active, out / "单品_活跃期统计.csv")
    write_csv(pairs, out / "单品对_overlap统计.csv")
    write_csv(item_stats, out / "单品_总体统计_重算.csv")
    write_csv(cat_stats, out / "品类_总体统计_重算.csv")
    item_compare = compare_existing(item, args.existing_dir / "单品_日销售汇总.csv", ["日期", "单品编码"], "当日正常销售量", "当日销售额")
    cat_compare = compare_existing(cat, args.existing_dir / "品类_日销售汇总.csv", ["日期", "分类编码"], "当日总销量", "当日销售额")
    write_csv(pd.DataFrame([{"table": "单品_日销售汇总.csv", **item_compare}, {"table": "品类_日销售汇总.csv", **cat_compare}]), out / "现有日表_核验结果.csv")
    markdown_report(args.report, product_audit, transaction_audit, item_compare, cat_compare, item, cat, calendar, active, pairs)
    print(json.dumps({"report": str(args.report), "output_dir": str(out), "item_rows": len(item), "category_rows": len(cat)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


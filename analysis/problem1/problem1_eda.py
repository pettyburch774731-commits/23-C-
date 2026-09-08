"""Reproducible exploratory analysis for 2023 CUMCM C, problem 1.

The script reads the question-1 CSV tables only.  It deliberately excludes the
ten calendar days with no sales records from daily panels, and it calculates
item-pair correlations only on dates where *both* items have a sales record.
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]


def choose_column(df: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Missing one of {candidates}; available columns: {list(df.columns)}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_item_daily(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        choose_column(df, "日期"): "date",
        choose_column(df, "单品编码"): "item_code",
        choose_column(df, "单品名称"): "item_name",
        choose_column(df, "分类编码"): "category_code",
        choose_column(df, "分类名称"): "category_name",
        choose_column(df, "销量", "当日正常销售量"): "quantity",
        choose_column(df, "销售额", "当日销售额"): "revenue",
        choose_column(df, "交易笔数", "当日交易笔数"): "transactions",
    }
    result = df.rename(columns=mapping)[list(mapping.values())].copy()
    result["date"] = pd.to_datetime(result["date"])
    return result


def normalize_category_daily(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        choose_column(df, "日期"): "date",
        choose_column(df, "分类编码"): "category_code",
        choose_column(df, "分类名称"): "category_name",
        choose_column(df, "当日总销量", "销量"): "quantity",
        choose_column(df, "当日销售额", "销售额"): "revenue",
        choose_column(df, "当日交易笔数", "交易笔数"): "transactions",
    }
    result = df.rename(columns=mapping)[list(mapping.values())].copy()
    result["date"] = pd.to_datetime(result["date"])
    return result


def setup_style() -> None:
    # Explicit registration avoids depending on a matplotlib font cache that may
    # not yet know about the Chinese fonts installed on Windows.
    chinese_font = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        chinese_name = font_manager.FontProperties(fname=str(chinese_font)).get_name()
    else:
        chinese_name = "Microsoft YaHei"
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": chinese_name,
        "font.sans-serif": [chinese_name, "Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig", float_format="%.6f")


def safe_cv(series: pd.Series) -> float:
    mean = series.mean()
    return float(series.std(ddof=1) / mean) if mean else np.nan


def category_panel(category_daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[pd.Timestamp]]:
    categories = category_daily[["category_code", "category_name"]].drop_duplicates().sort_values("category_code")
    names = categories["category_name"].tolist()
    observed_dates = sorted(category_daily["date"].drop_duplicates())
    raw = category_daily.pivot(index="date", columns="category_name", values="quantity")
    # A date in observed_dates has at least one store sales record.  A missing
    # category on that date therefore means zero category sales; dates absent
    # from observed_dates are intentionally not introduced here.
    panel = raw.reindex(index=observed_dates, columns=names).fillna(0.0)
    return panel, names, observed_dates


def category_analysis(category_daily: pd.DataFrame, outputs: Path, figures: Path) -> dict:
    panel, names, observed_dates = category_panel(category_daily)
    stat_rows = []
    for name in names:
        values = panel[name]
        stat_rows.append({
            "分类名称": name,
            "总销量_kg": values.sum(),
            "观测日数": len(values),
            "日均销量_kg": values.mean(),
            "中位数_kg": values.median(),
            "标准差_kg": values.std(ddof=1),
            "CV": safe_cv(values),
            "Q1_kg": values.quantile(0.25),
            "Q3_kg": values.quantile(0.75),
            "偏度": values.skew(),
            "零销量观测日数": int((values == 0).sum()),
        })
    stats = pd.DataFrame(stat_rows).sort_values("总销量_kg", ascending=False)
    write_csv(stats, outputs / "category" / "品类_描述统计.csv")

    monthly = panel.copy()
    monthly["月份"] = monthly.index.month
    month_avg = monthly.groupby("月份")[names].mean().reset_index()
    write_csv(month_avg, outputs / "category" / "品类_月度日均销量.csv")

    # Keep years separate so an apparent pooled month pattern can be checked for
    # repetition across the three observed years.
    ym = panel.copy()
    ym["年份"] = ym.index.year
    ym["月份"] = ym.index.month
    year_month_avg = ym.groupby(["年份", "月份"])[names].mean().reset_index()
    write_csv(year_month_avg, outputs / "category" / "品类_跨年度_月度日均销量.csv")
    # Only 2021 and 2022 contain all twelve months. Compare their curves
    # directly; the half-years 2020/2023 remain visual context, not replicates.
    full_2021 = year_month_avg[year_month_avg["年份"] == 2021].sort_values("月份").reset_index(drop=True)
    full_2022 = year_month_avg[year_month_avg["年份"] == 2022].sort_values("月份").reset_index(drop=True)
    stability_rows = []
    for name in names:
        peak_21, peak_22 = int(full_2021.loc[full_2021[name].idxmax(), "月份"]), int(full_2022.loc[full_2022[name].idxmax(), "月份"])
        trough_21, trough_22 = int(full_2021.loc[full_2021[name].idxmin(), "月份"]), int(full_2022.loc[full_2022[name].idxmin(), "月份"])
        pearson = full_2021[name].corr(full_2022[name], method="pearson")
        spearman = full_2021[name].corr(full_2022[name], method="spearman")
        peak_close = abs(peak_21 - peak_22) <= 1
        trough_close = abs(trough_21 - trough_22) <= 1
        # This conservative label is an evidence summary, not a claim that
        # seasonality has been proved from only two complete years.
        label = "完整年度间有一定重复线索" if pearson >= .5 and spearman >= .5 and peak_close and trough_close else "跨完整年度不稳定，仅作描述"
        stability_rows.append({"分类名称": name, "2021_峰值月份": peak_21, "2022_峰值月份": peak_22, "峰值月份相近_≤1月": peak_close,
                               "2021_低谷月份": trough_21, "2022_低谷月份": trough_22, "低谷月份相近_≤1月": trough_close,
                               "2021_2022月均曲线Pearson": pearson, "2021_2022月均曲线Spearman": spearman, "判定": label})
    seasonal_stability = pd.DataFrame(stability_rows)
    write_csv(seasonal_stability, outputs / "category" / "品类_2021_2022月份规律一致性.csv")

    weekdays = panel.copy()
    weekdays["星期"] = weekdays.index.dayofweek
    weekday_avg = weekdays.groupby("星期")[names].mean().reindex(range(7)).reset_index()
    weekday_avg["星期名称"] = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    write_csv(weekday_avg, outputs / "category" / "品类_星期日均销量.csv")

    calendar_days = pd.Series((panel.index - panel.index.min()).days.astype(float), index=panel.index)
    trend = pd.DataFrame({
        "分类名称": names,
        "Spearman_距起始日期日历天数": [panel[name].corr(calendar_days, method="spearman") for name in names],
    })
    write_csv(trend, outputs / "category" / "品类_长期趋势统计.csv")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = stats.sort_values("总销量_kg")
    ax.barh(ordered["分类名称"], ordered["总销量_kg"], color=sns.color_palette("Blues_d", len(ordered)))
    ax.set(xlabel="三年总销量（kg）", ylabel="")
    ax.set_title("六类蔬菜三年总销量")
    fig.savefig(figures / "品类_总销量柱状图.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.boxplot(data=panel[names], ax=ax, showfliers=False)
    ax.set(title="品类日销量分布（仅含有销售记录日）", xlabel="", ylabel="日销量（kg）")
    ax.tick_params(axis="x", rotation=18)
    fig.savefig(figures / "品类_日销量箱线图.png")
    plt.close(fig)

    # A time offset makes this a true calendar window. Missing store-record days
    # are absent rather than zero-filled, so each mean uses only valid dates in
    # the preceding 28 calendar days.
    rolling = panel.rolling("28D", min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for name in names:
        ax.plot(rolling.index, rolling[name], linewidth=1.2, label=name)
    ax.set(title="品类日销量：28 日历日窗口内有效观测日平均", xlabel="日期", ylabel="日销量（kg）")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(figures / "品类_时间规律_28日移动平均.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in names:
        ax.plot(month_avg["月份"], month_avg[name], marker="o", linewidth=1.3, label=name)
    ax.set(title="按自然月汇总的品类平均日销量", xlabel="月份", ylabel="平均日销量（kg）", xticks=range(1, 13))
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(figures / "品类_月度日均销量.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True)
    for ax, name in zip(axes.flat, names):
        for year, part in year_month_avg.groupby("年份"):
            ax.plot(part["月份"], part[name], marker="o", linewidth=1.1, label=str(year))
        ax.set(title=name, xlabel="月份", ylabel="平均日销量（kg）", xticks=range(1, 13))
        ax.legend(fontsize=7)
    fig.suptitle("品类月度规律的跨年度验证", y=1.02)
    fig.tight_layout()
    fig.savefig(figures / "品类_跨年度月度规律.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(7)
    width = 0.12
    for i, name in enumerate(names):
        ax.bar(x + (i - 2.5) * width, weekday_avg[name], width=width, label=name)
    ax.set(title="不同星期的品类平均日销量", xlabel="星期", ylabel="平均日销量（kg）", xticks=x, xticklabels=weekday_avg["星期名称"])
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(figures / "品类_星期日均销量.png")
    plt.close(fig)

    return {"panel": panel, "stats": stats, "monthly": month_avg, "year_month": year_month_avg, "seasonal_stability": seasonal_stability, "weekday": weekday_avg, "trend": trend, "observed_days": len(observed_dates)}


def item_analysis(item_daily: pd.DataFrame, active: pd.DataFrame | None, outputs: Path, figures: Path) -> dict:
    group = item_daily.groupby(["item_code", "item_name", "category_code", "category_name"], as_index=False)
    stats = group.agg(
        总销量_kg=("quantity", "sum"),
        总销售额_元=("revenue", "sum"),
        有销售日数=("date", "nunique"),
        有销量日均销量_kg=("quantity", "mean"),
        日销量中位数_kg=("quantity", "median"),
        日销量标准差_kg=("quantity", "std"),
        最大日销量_kg=("quantity", "max"),
    )
    stats = stats.rename(columns={"item_code": "单品编码", "item_name": "单品名称", "category_code": "分类编码", "category_name": "分类名称"})
    stats["CV_有销量日"] = stats["日销量标准差_kg"] / stats["有销量日均销量_kg"]
    if active is not None:
        active = active.rename(columns={
            choose_column(active, "单品编码"): "item_code",
            choose_column(active, "first_sale_date"): "首次销售日期",
            choose_column(active, "last_sale_date"): "最后销售日期",
            choose_column(active, "sales_days"): "销售天数_审计",
            choose_column(active, "active_span_calendar_days"): "活跃跨度_日历日",
            choose_column(active, "active_span_observed_sales_days"): "活跃跨度_有销售记录日",
            choose_column(active, "sales_days_share_of_active_span"): "销售天数占活跃跨度比例",
        })
        active["首次销售日期"] = pd.to_datetime(active["首次销售日期"])
        active["最后销售日期"] = pd.to_datetime(active["最后销售日期"])
        cols = ["item_code", "首次销售日期", "最后销售日期", "活跃跨度_日历日", "活跃跨度_有销售记录日", "销售天数占活跃跨度比例"]
        active = active.rename(columns={"item_code": "单品编码"})
        cols[0] = "单品编码"
        stats = stats.merge(active[cols], on="单品编码", how="left")
    stats = stats.sort_values("总销量_kg", ascending=False).reset_index(drop=True)
    stats["销量排名"] = np.arange(1, len(stats) + 1)
    stats["累计销量占比"] = stats["总销量_kg"].cumsum() / stats["总销量_kg"].sum()
    write_csv(stats, outputs / "item" / "单品_描述统计与排名.csv")
    write_csv(stats.head(20), outputs / "item" / "单品_Top20.csv")

    summary = pd.DataFrame([{
        "单品数": len(stats),
        "总销量_kg": stats["总销量_kg"].sum(),
        "销量中位数_kg": stats["总销量_kg"].median(),
        "销量Q1_kg": stats["总销量_kg"].quantile(.25),
        "销量Q3_kg": stats["总销量_kg"].quantile(.75),
        "Top10销量占比": stats.head(10)["总销量_kg"].sum() / stats["总销量_kg"].sum(),
        "Top20销量占比": stats.head(20)["总销量_kg"].sum() / stats["总销量_kg"].sum(),
        "销售天数中位数": stats["有销售日数"].median(),
        "CV中位数": stats["CV_有销量日"].median(),
    }])
    write_csv(summary, outputs / "item" / "单品_分布摘要.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(stats["总销量_kg"], bins=35, color="#4C78A8", edgecolor="white")
    axes[0].set(title="单品总销量分布", xlabel="三年总销量（kg）", ylabel="单品数")
    axes[1].hist(np.log1p(stats["总销量_kg"]), bins=35, color="#72B7B2", edgecolor="white")
    axes[1].set(title="单品总销量分布（log1p 刻度）", xlabel="log(1+总销量)", ylabel="单品数")
    fig.savefig(figures / "单品_总销量分布.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = stats["销量排名"]
    ax1.plot(x, stats["总销量_kg"], color="#4C78A8", linewidth=1.3)
    ax1.set(xlabel="单品销量排名", ylabel="单品总销量（kg）", title="单品销量排名与累计贡献")
    ax2 = ax1.twinx()
    ax2.plot(x, stats["累计销量占比"] * 100, color="#E45756", linewidth=1.5)
    ax2.set_ylabel("累计销量占比（%）")
    fig.savefig(figures / "单品_销量排名与累计贡献.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(stats["有销售日数"], bins=30, color="#54A24B", edgecolor="white")
    axes[0].set(title="单品有销售日数分布", xlabel="有销售日数", ylabel="单品数")
    valid_cv = stats["CV_有销量日"].replace([np.inf, -np.inf], np.nan).dropna()
    axes[1].hist(valid_cv.clip(upper=valid_cv.quantile(.99)), bins=30, color="#F58518", edgecolor="white")
    axes[1].set(title="单品有销量日 CV 分布（99% 截尾展示）", xlabel="CV", ylabel="单品数")
    fig.savefig(figures / "单品_销售天数与波动分布.png")
    plt.close(fig)

    return {"stats": stats, "summary": summary}


def heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, ax=ax)
    ax.set(title=title, xlabel="", ylabel="")
    ax.tick_params(axis="x", rotation=28)
    ax.tick_params(axis="y", rotation=0)
    fig.savefig(path)
    plt.close(fig)


def residualize_time_effects(panel: pd.DataFrame, include_year: bool = False) -> pd.DataFrame:
    """Remove additive weekday, calendar-month and linear-time effects by OLS.

    This is a transparent descriptive adjustment, not a causal model.  The time
    coordinate uses actual calendar-day distance, so the ten unobserved dates do
    not compress the secular trend.
    """
    dates = panel.index
    design = pd.DataFrame({"intercept": 1.0, "time_days": (dates - dates.min()).days.astype(float)}, index=dates)
    weekday = pd.get_dummies(dates.dayofweek, prefix="weekday", drop_first=True, dtype=float)
    month = pd.get_dummies(dates.month, prefix="month", drop_first=True, dtype=float)
    weekday.index = dates
    month.index = dates
    parts = [design, weekday, month]
    if include_year:
        year = pd.get_dummies(dates.year, prefix="year", drop_first=True, dtype=float)
        year.index = dates
        parts.append(year)
    x = pd.concat(parts, axis=1).to_numpy(dtype=float)
    residuals = pd.DataFrame(index=dates, columns=panel.columns, dtype=float)
    for name in panel.columns:
        y = panel[name].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals[name] = y - x @ beta
    return residuals


def category_correlations(panel: pd.DataFrame, outputs: Path, figures: Path) -> dict:
    daily_p = panel.corr(method="pearson")
    daily_s = panel.corr(method="spearman")

    # The previous resample sum retained partial first/last weeks and weeks with
    # unobserved store dates.  A W-SUN period is Monday--Sunday; exactly seven
    # observed rows is therefore a complete natural week with sales records.
    weekly_all = panel.resample("W-SUN").sum()
    weekly_all_p = weekly_all.corr(method="pearson")
    weekly_all_s = weekly_all.corr(method="spearman")
    natural_week = panel.index.to_period("W-SUN")
    weekly_sum = panel.groupby(natural_week).sum()
    weekly_count = pd.Series(1, index=panel.index).groupby(natural_week).sum()
    weekly = weekly_sum.loc[weekly_count[weekly_count == 7].index]
    weekly_p = weekly.corr(method="pearson")
    weekly_s = weekly.corr(method="spearman")

    residuals = residualize_time_effects(panel)
    residual_p = residuals.corr(method="pearson")
    residual_s = residuals.corr(method="spearman")
    residuals_year = residualize_time_effects(panel, include_year=True)
    residual_year_p = residuals_year.corr(method="pearson")
    residual_year_s = residuals_year.corr(method="spearman")
    for name, table in [("品类_日尺度_Pearson相关矩阵.csv", daily_p), ("品类_日尺度_Spearman相关矩阵.csv", daily_s),
                        ("品类_周尺度_旧含不完整周_Pearson相关矩阵.csv", weekly_all_p), ("品类_周尺度_旧含不完整周_Spearman相关矩阵.csv", weekly_all_s),
                        ("品类_周尺度_完整自然周_Pearson相关矩阵.csv", weekly_p), ("品类_周尺度_完整自然周_Spearman相关矩阵.csv", weekly_s),
                        ("品类_残差_Pearson相关矩阵.csv", residual_p), ("品类_残差_Spearman相关矩阵.csv", residual_s),
                        ("品类_残差_含年份效应_Pearson相关矩阵.csv", residual_year_p), ("品类_残差_含年份效应_Spearman相关矩阵.csv", residual_year_s)]:
        write_csv(table.reset_index().rename(columns={"category_name": "分类名称"}), outputs / "correlation" / name)
    heatmap(daily_p, "品类日销量 Pearson 相关", figures / "品类_日尺度Pearson热力图.png")
    heatmap(daily_s, "品类日销量 Spearman 相关", figures / "品类_日尺度Spearman热力图.png")
    heatmap(weekly_p, "品类完整自然周销量 Pearson 相关", figures / "品类_周尺度完整自然周Pearson热力图.png")
    heatmap(weekly_s, "品类完整自然周销量 Spearman 相关", figures / "品类_周尺度完整自然周Spearman热力图.png")
    heatmap(residual_p, "控制星期、月份、线性时间趋势后的 Pearson 残差相关", figures / "品类_残差Pearson热力图.png")
    heatmap(residual_s, "控制星期、月份、线性时间趋势后的 Spearman 残差相关", figures / "品类_残差Spearman热力图.png")

    pairs = []
    for a, b in itertools.combinations(panel.columns, 2):
        pairs.append({"品类A": a, "品类B": b, "日Pearson": daily_p.loc[a, b], "日Spearman": daily_s.loc[a, b],
                      "旧周Pearson": weekly_all_p.loc[a, b], "旧周Spearman": weekly_all_s.loc[a, b],
                      "完整周Pearson": weekly_p.loc[a, b], "完整周Spearman": weekly_s.loc[a, b],
                      "残差Pearson": residual_p.loc[a, b], "残差Spearman": residual_s.loc[a, b],
                      "残差含年份Pearson": residual_year_p.loc[a, b], "残差含年份Spearman": residual_year_s.loc[a, b]})
    pair_table = pd.DataFrame(pairs)
    pair_table["周Pearson修正差异"] = pair_table["完整周Pearson"] - pair_table["旧周Pearson"]
    pair_table["周Spearman修正差异"] = pair_table["完整周Spearman"] - pair_table["旧周Spearman"]
    pair_table["残差Pearson_年份效应差异"] = pair_table["残差含年份Pearson"] - pair_table["残差Pearson"]
    pair_table["残差Spearman_年份效应差异"] = pair_table["残差含年份Spearman"] - pair_table["残差Spearman"]
    pair_table["最大绝对相关"] = pair_table[["日Pearson", "日Spearman", "完整周Pearson", "完整周Spearman", "残差Pearson", "残差Spearman", "残差含年份Pearson", "残差含年份Spearman"]].abs().max(axis=1)
    pair_table = pair_table.sort_values("最大绝对相关", ascending=False)
    write_csv(pair_table, outputs / "correlation" / "品类_相关性_时间控制与周修正对照.csv")
    focus_names = [("辣椒类", "食用菌"), ("花叶类", "花菜类"), ("水生根茎类", "食用菌")]
    focus = pair_table.set_index(["品类A", "品类B"])
    focus_rows = [{"品类A": pair[0], "品类B": pair[1], **focus.loc[pair].to_dict()} for pair in focus_names]
    focus_table = pd.DataFrame(focus_rows)
    write_csv(focus_table, outputs / "correlation" / "品类_重点对_控制时间效应对照.csv")
    week_summary = pd.DataFrame([{"旧周样本数": len(weekly_all), "完整自然周样本数": len(weekly), "剔除周数": len(weekly_all) - len(weekly)}])
    write_csv(week_summary, outputs / "correlation" / "品类_周尺度样本修正摘要.csv")
    return {"daily_p": daily_p, "daily_s": daily_s, "weekly_old_p": weekly_all_p, "weekly_old_s": weekly_all_s, "weekly_p": weekly_p, "weekly_s": weekly_s, "residual_p": residual_p, "residual_s": residual_s, "residual_year_p": residual_year_p, "residual_year_s": residual_year_s, "pairs": pair_table, "focus": focus_table, "weekly": weekly, "weekly_old": weekly_all, "residuals": residuals, "residuals_year": residuals_year}


def item_overlap_and_correlations(item_daily: pd.DataFrame, active: pd.DataFrame | None, outputs: Path, figures: Path) -> dict:
    values = {code: group.set_index("date")["quantity"].sort_index() for code, group in item_daily.groupby("item_code")}
    meta = item_daily[["item_code", "item_name", "category_name"]].drop_duplicates("item_code").set_index("item_code")
    overlap_rows = []
    for a, b in itertools.combinations(sorted(values), 2):
        common = values[a].index.intersection(values[b].index)
        overlap_rows.append({"单品编码_A": a, "单品编码_B": b, "共同有销售记录的日期数": len(common)})
    overlap = pd.DataFrame(overlap_rows)
    q = overlap["共同有销售记录的日期数"].quantile([.1, .25, .5, .75, .9, .95, .99])
    # Thirty dates is above the empirical third quartile (18 days) and supplies
    # roughly a month of joint observations.  The 20/60 counts are retained for
    # transparent sensitivity context; correlations use 30 as the primary rule.
    threshold = 30
    overlap_summary = pd.DataFrame([
        {"单品对数": len(overlap), "最小值": overlap.iloc[:, 2].min(), "Q1": q.loc[.25], "中位数": q.loc[.5],
         "Q3": q.loc[.75], "P90": q.loc[.9], "P95": q.loc[.95], "P99": q.loc[.99], "最大值": overlap.iloc[:, 2].max(),
         "共同销售日≥20的单品对数": int((overlap.iloc[:, 2] >= 20).sum()),
         "共同销售日≥30的单品对数": int((overlap.iloc[:, 2] >= 30).sum()),
         "共同销售日≥60的单品对数": int((overlap.iloc[:, 2] >= 60).sum()),
         "主分析最低共同销售日门槛": threshold}
    ])
    write_csv(overlap_summary, outputs / "correlation" / "单品_overlap分布摘要.csv")
    # Commit a compact distribution table instead of the 30,135-row working table.
    bins = [-1, 0, 1, 7, 18, 29, 59, 119, np.inf]
    labels = ["0", "1", "2–7", "8–18", "19–29", "30–59", "60–119", "≥120"]
    distribution = pd.cut(overlap.iloc[:, 2], bins=bins, labels=labels).value_counts().reindex(labels).rename_axis("共同有销售日区间").reset_index(name="单品对数")
    distribution["占比"] = distribution["单品对数"] / len(overlap)
    write_csv(distribution, outputs / "correlation" / "单品_overlap分布区间.csv")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(distribution["共同有销售日区间"], distribution["单品对数"], color="#4C78A8")
    ax.set(title="单品对共同有销售记录日期的分布", xlabel="共同有销售日", ylabel="单品对数")
    fig.savefig(figures / "单品_overlap分布.png")
    plt.close(fig)

    eligible = overlap[overlap.iloc[:, 2] >= threshold]
    corr_rows = []
    for row in eligible.itertuples(index=False):
        a, b, n = row
        joined = pd.concat([values[a], values[b]], axis=1, join="inner")
        joined.columns = ["quantity_a", "quantity_b"]
        if joined["quantity_a"].nunique() < 2 or joined["quantity_b"].nunique() < 2:
            continue
        corr_rows.append({
            "单品编码_A": a, "单品名称_A": meta.loc[a, "item_name"], "分类名称_A": meta.loc[a, "category_name"],
            "单品编码_B": b, "单品名称_B": meta.loc[b, "item_name"], "分类名称_B": meta.loc[b, "category_name"],
            "共同有销售记录日数": n,
            "Pearson相关": joined["quantity_a"].corr(joined["quantity_b"], method="pearson"),
            "Spearman相关": joined["quantity_a"].corr(joined["quantity_b"], method="spearman"),
        })
    correlations = pd.DataFrame(corr_rows)
    correlations["最大绝对相关"] = correlations[["Pearson相关", "Spearman相关"]].abs().max(axis=1)
    correlations["Pearson_Spearman绝对差"] = (correlations["Pearson相关"] - correlations["Spearman相关"]).abs()
    correlations["Pearson_Spearman同号"] = correlations["Pearson相关"] * correlations["Spearman相关"] > 0
    correlations_30 = correlations.copy()
    correlations_60 = correlations[correlations["共同有销售记录日数"] >= 60].copy()
    pos30 = correlations_30.sort_values("Pearson相关", ascending=False).head(20)
    neg30 = correlations_30.sort_values("Pearson相关", ascending=True).head(20)
    pos60 = correlations_60.sort_values("Pearson相关", ascending=False).head(20)
    neg60 = correlations_60.sort_values("Pearson相关", ascending=True).head(20)
    write_csv(pos30, outputs / "correlation" / "单品_n不少于30_正相关Top20.csv")
    write_csv(neg30, outputs / "correlation" / "单品_n不少于30_负相关Top20.csv")
    write_csv(pos60, outputs / "correlation" / "单品_n不少于60_正相关Top20.csv")
    write_csv(neg60, outputs / "correlation" / "单品_n不少于60_负相关Top20.csv")
    write_csv(correlations.sort_values("Pearson_Spearman绝对差", ascending=False).head(50), outputs / "correlation" / "单品_Pearson与Spearman差异Top50.csv")

    # A pair is called robust only when it survives the larger-sample rule,
    # both correlations have the same sign, and both reach moderate strength.
    # The 0.40 cutoff is deliberately applied to the weaker of the two metrics.
    robust = correlations_60[(correlations_60["Pearson_Spearman同号"]) &
                             (correlations_60[["Pearson相关", "Spearman相关"]].abs().min(axis=1) >= .40)].copy()
    robust["共同最弱相关绝对值"] = robust[["Pearson相关", "Spearman相关"]].abs().min(axis=1)
    robust = robust.sort_values("共同最弱相关绝对值", ascending=False)
    write_csv(robust, outputs / "correlation" / "单品_稳健相关对_n不少于60.csv")
    top30_abs = correlations_30.assign(排名依据=correlations_30["最大绝对相关"]).sort_values("排名依据", ascending=False).head(20)
    top60_abs = correlations_60.assign(排名依据=correlations_60["最大绝对相关"]).sort_values("排名依据", ascending=False).head(20)
    key_cols = ["单品编码_A", "单品编码_B"]
    common_top_abs = pd.merge(top30_abs[key_cols], top60_abs[key_cols], on=key_cols)
    sensitivity = pd.DataFrame([{
        "n≥30可计算单品对数": len(correlations_30), "n≥60可计算单品对数": len(correlations_60),
        "n≥30绝对相关Top20与n≥60绝对相关Top20交集数": len(common_top_abs),
        "n≥30 Pearson与Spearman异号对数": int((~correlations_30["Pearson_Spearman同号"]).sum()),
        "n≥60 Pearson与Spearman异号对数": int((~correlations_60["Pearson_Spearman同号"]).sum()),
        "稳健相关对数_n≥60_同号且两者绝对值≥0.40": len(robust),
    }])
    write_csv(sensitivity, outputs / "correlation" / "单品_相关性门槛敏感性摘要.csv")
    write_csv(correlations.sort_values("最大绝对相关", ascending=False).head(100), outputs / "correlation" / "单品_相关性绝对值Top100.csv")

    # Do not present hundreds of qualifying pairs as a flat list.  Summarize the
    # relation structure and select examples with both adequate overlap and close
    # Pearson/Spearman estimates.
    robust = robust.copy()
    robust["方向"] = np.where(robust["Pearson相关"] > 0, "正相关", "负相关")
    robust["是否同品类"] = robust["分类名称_A"] == robust["分类名称_B"]
    robust["品类组合"] = robust.apply(lambda r: " × ".join(sorted([r["分类名称_A"], r["分类名称_B"]])), axis=1)
    structure_summary = pd.DataFrame([{
        "稳健相关对总数": len(robust), "正相关对数": int((robust["方向"] == "正相关").sum()), "负相关对数": int((robust["方向"] == "负相关").sum()),
        "同品类对数": int(robust["是否同品类"].sum()), "跨品类对数": int((~robust["是否同品类"]).sum()),
    }])
    write_csv(structure_summary, outputs / "correlation" / "单品_稳健相关对_结构摘要.csv")
    structure_by_category = robust.groupby(["品类组合", "方向"], as_index=False).agg(
        稳健相关对数=("单品编码_A", "size"), 共同销售日数中位数=("共同有销售记录日数", "median"),
        Pearson中位数=("Pearson相关", "median"), Spearman中位数=("Spearman相关", "median"),
        共同最弱相关绝对值中位数=("共同最弱相关绝对值", "median"),
    ).sort_values(["稳健相关对数", "共同最弱相关绝对值中位数"], ascending=False)
    write_csv(structure_by_category, outputs / "correlation" / "单品_稳健相关对_品类组合汇总.csv")
    n_cutoff = robust["共同有销售记录日数"].quantile(.75)
    representative_pool = robust[(robust["共同有销售记录日数"] >= n_cutoff) & (robust["Pearson_Spearman绝对差"] <= .10)].copy()
    representative_pool["代表性排序分数"] = representative_pool["共同最弱相关绝对值"] * np.log1p(representative_pool["共同有销售记录日数"])
    representatives = pd.concat([
        representative_pool[representative_pool["方向"] == "正相关"].nlargest(3, "代表性排序分数"),
        representative_pool[representative_pool["方向"] == "负相关"].nlargest(2, "代表性排序分数"),
    ]).sort_values(["方向", "代表性排序分数"], ascending=[False, False])
    write_csv(representatives, outputs / "correlation" / "单品_稳健相关对_代表性样本.csv")

    # Scatter examples use the robust set if it contains both signs.  Raw top
    # Pearson pairs remain tables only because a high Pearson alone is a clue.
    examples = []
    if not robust.empty:
        robust_pos = robust[robust["Pearson相关"] > 0]
        robust_neg = robust[robust["Pearson相关"] < 0]
        if not robust_pos.empty:
            examples.append(("稳健正", robust_pos.iloc[0]))
        if not robust_neg.empty:
            examples.append(("稳健负", robust_neg.iloc[0]))
    for label, row in examples:
        joined = pd.concat([values[row["单品编码_A"]], values[row["单品编码_B"]]], axis=1, join="inner")
        joined.columns = [row["单品名称_A"], row["单品名称_B"]]
        fig, ax = plt.subplots(figsize=(5.5, 4.7))
        sns.regplot(data=joined, x=row["单品名称_A"], y=row["单品名称_B"], scatter_kws={"s": 15, "alpha": .55}, line_kws={"color": "#E45756"}, ax=ax)
        ax.set(title=f"单品共同销售日数量关系：{label}相关\nPearson={row['Pearson相关']:.2f}, n={int(row['共同有销售记录日数'])}")
        fig.savefig(figures / f"单品_{label}相关_散点图.png")
        plt.close(fig)

    return {"overlap_summary": overlap_summary, "distribution": distribution, "threshold": threshold, "eligible_pairs": len(eligible),
            "correlations": correlations, "correlations_30": correlations_30, "correlations_60": correlations_60,
            "positive_30": pos30, "negative_30": neg30, "positive_60": pos60, "negative_60": neg60,
            "robust": robust, "sensitivity": sensitivity, "structure_summary": structure_summary,
            "structure_by_category": structure_by_category, "representatives": representatives, "representative_n_cutoff": n_cutoff}


def fmt(value: float, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    selected = df[columns].head(n) if n else df[columns]
    return selected.to_markdown(index=False, floatfmt=".2f")


def write_final_candidate_tables(category: dict, corr: dict, item_corr: dict, outputs: Path) -> dict:
    """Write the shared missing-data rule and the paper-oriented evidence tables."""
    final_dir = outputs / "final"
    rules = pd.DataFrame([
        {"统计问题": "三年总销量", "有效样本": "所有真实销售记录", "缺失处理": "直接求和；无销售记录日不另造 0", "完整窗口要求": "否"},
        {"统计问题": "日均销量", "有效样本": "1085 个有销售记录日", "缺失处理": "仅这些有效日参与均值", "完整窗口要求": "否"},
        {"统计问题": "月度平均日销量", "有效样本": "该月份的所有有效观测日", "缺失处理": "不要求该月完整，不删除相邻有效日", "完整窗口要求": "否"},
        {"统计问题": "跨年度月份规律", "有效样本": "各 year-month 内的所有有效观测日", "缺失处理": "按 year-month 独立平均；半年度不当完整重复", "完整窗口要求": "否"},
        {"统计问题": "星期规律", "有效样本": "所有有效周一、周二……", "缺失处理": "按星期分别平均，不因同周其他日缺失而删除", "完整窗口要求": "否"},
        {"统计问题": "日尺度相关", "有效样本": "有效观测日", "缺失处理": "不把全店无销售记录日补为 0", "完整窗口要求": "否"},
        {"统计问题": "周总销量及周尺度相关", "有效样本": "周一至周日均有销售记录的周", "缺失处理": "仅此处剔除 7 日不齐全的自然周", "完整窗口要求": "是：7 个销售记录日"},
        {"统计问题": "单品相关", "有效样本": "两单品共同销售日", "缺失处理": "未知状态日期不补 0", "完整窗口要求": "否；使用 overlap 门槛"},
    ])
    write_csv(rules, final_dir / "统计口径与缺失处理.csv")
    focus = corr["focus"].set_index(["品类A", "品类B"])
    structure = item_corr["structure_summary"].iloc[0]
    monthly = category["seasonal_stability"]
    conclusion = pd.DataFrame([
        {"结论": "品类规模和波动存在差异", "使用指标/图": "品类描述统计、总销量柱状图、箱线图", "数值证据": "花叶类总销量最高；水生根茎类 CV 最高", "稳健性验证": "1085 个有效日，未把 10 个无记录日补零", "论文表述限制": "描述性差异，不解释原因"},
        {"结论": "月度模式跨完整年度总体不稳定", "使用指标/图": "2021/2022 月均曲线与一致性表", "数值证据": f"6 类中满足保守重复规则的类别数={int((monthly['判定'] == '完整年度间有一定重复线索').sum())}", "稳健性验证": "仅比较完整 2021、2022；2020/2023 仅作半年度旁证", "论文表述限制": "不能称已证明稳定季节性"},
        {"结论": "辣椒类—食用菌存在较强的时间控制后同步线索", "使用指标/图": "日相关、完整周相关、残差相关", "数值证据": f"残差 Pearson/Spearman={focus.loc[('辣椒类','食用菌'),'残差Pearson']:.3f}/{focus.loc[('辣椒类','食用菌'),'残差Spearman']:.3f}；含年份={focus.loc[('辣椒类','食用菌'),'残差含年份Pearson']:.3f}/{focus.loc[('辣椒类','食用菌'),'残差含年份Spearman']:.3f}", "稳健性验证": "完整自然周与年份效应敏感性", "论文表述限制": "同步变化，不是因果"},
        {"结论": "花叶类—花菜类与水生根茎类—食用菌仍有正向同步线索", "使用指标/图": "重点对照表与残差热图", "数值证据": "残差相关方向为正；见重点对照表", "稳健性验证": "完整周、含年份效应残差", "论文表述限制": "强度不同，不能外推为机制"},
        {"结论": "单品关系需使用 overlap 与双相关筛选", "使用指标/图": "overlap 分布、门槛敏感性、稳健相关结构表", "数值证据": f"n≥30/60 Top20 交集=11；稳健对={int(structure['稳健相关对总数'])}，正/负={int(structure['正相关对数'])}/{int(structure['负相关对数'])}", "稳健性验证": "n≥60、Pearson/Spearman 同号且均≥0.40", "论文表述限制": "仅共同销售日线索；不证明替代或互补"},
    ])
    write_csv(conclusion, final_dir / "结论_证据_限制.csv")
    candidate_figures = pd.DataFrame([
        {"图或表": "品类总销量柱状图、日销量箱线图", "建议位置": "正文", "用途": "品类规模与波动"},
        {"图或表": "跨年度月度规律图", "建议位置": "正文", "用途": "月度规律及其跨年限制"},
        {"图或表": "完整自然周 Pearson/Spearman 热图", "建议位置": "正文", "用途": "周尺度品类同步关系"},
        {"图或表": "残差 Pearson 热图", "建议位置": "正文", "用途": "控制时间规律后的品类关系"},
        {"图或表": "单品销量排名与累计贡献、overlap 分布", "建议位置": "正文", "用途": "集中度与单品关系口径"},
        {"图或表": "稳健单品对代表性样本表", "建议位置": "正文或附表", "用途": "少量可解释的单品关系线索"},
        {"图或表": "旧含不完整周相关、全量 Top20/Top100、Pearson-Spearman 差异表", "建议位置": "分析目录/附录", "用途": "口径修正与稳健性审计，不作论文主结论"},
    ])
    write_csv(candidate_figures, final_dir / "论文候选图表清单.csv")
    return {"rules": rules, "conclusion": conclusion, "figures": candidate_figures}


def create_report(category: dict, item: dict, corr: dict, item_corr: dict, report_path: Path) -> None:
    cat_stats = category["stats"]
    item_stats = item["stats"]
    item_summary = item["summary"].iloc[0]
    cp = corr["pairs"].iloc[0]
    pos = item_corr["positive"].iloc[0]
    neg = item_corr["negative"].iloc[0]
    trend = category["trend"].sort_values("Spearman_日期趋势", key=lambda s: s.abs(), ascending=False)
    report = f"""# 2023 年 CUMCM C 题：问题一 EDA（第一版）

> **范围。** 本报告只做问题一的统计描述、时间规律和相关关系探索，不包含预测、补货、定价或问题二、问题三。所有结果由 `problem1_eda.py` 从问题一 CSV 直接生成。

## 数据与口径

- 输入为 `问题1_建模数据/单品_日销售汇总.csv`、`品类_日销售汇总.csv`，并使用审计输出的单品活跃期统计辅助解释单品覆盖期。
- 日尺度品类面板包含 **{category['observed_days']} 个有销售记录日**。完整三年日历中的 10 个“无销售记录日”没有被填为零或纳入日均值、相关性计算。
- 在有销售记录的某天，某一品类未出现于流水，可视为该品类当天销量为 0；单品层面则不将缺失销售记录自动补成 0。
- 单品相关性使用两种单品均有销售记录的共同日期，描述共同销售日上的销量同步变化，不能解释为全部在售日的需求相关性，更不能作为因果证据。

## A. 品类销售量分布与时间规律

### 统计量回答的问题

- **总销量**比较三年销售规模；**日均值、中位数、四分位数**描述典型日水平和分布范围。
- **标准差、CV（标准差/日均值）**衡量绝对波动和相对波动；**偏度**提示是否由少数高销量日拉长右尾。
- **日期 Spearman 相关**只用作单调长期趋势的初步线索，不能替代季节分解或预测模型。

### 六品类描述统计

{markdown_table(cat_stats, ['分类名称','总销量_kg','日均销量_kg','中位数_kg','标准差_kg','CV','Q1_kg','Q3_kg','偏度'])}

总销量最高的是 **{cat_stats.iloc[0]['分类名称']}**（{cat_stats.iloc[0]['总销量_kg']:.1f} kg），最低的是 **{cat_stats.iloc[-1]['分类名称']}**（{cat_stats.iloc[-1]['总销量_kg']:.1f} kg）。相对波动最大的品类为 **{cat_stats.loc[cat_stats['CV'].idxmax(), '分类名称']}**（CV={cat_stats['CV'].max():.2f}），最稳定的为 **{cat_stats.loc[cat_stats['CV'].idxmin(), '分类名称']}**（CV={cat_stats['CV'].min():.2f}）。

长期单调变化线索中，绝对 Spearman 值最大的品类是 **{trend.iloc[0]['分类名称']}**（{trend.iloc[0]['Spearman_日期趋势']:.2f}）；需与移动平均图、月度图共同阅读，不能据此直接认定趋势机制。

主要图：

- `figures/品类_总销量柱状图.png`：比较六类销售规模。
- `figures/品类_日销量箱线图.png`：比较日销量中心位置与离散度，异常点未为展示而主导图形。
- `figures/品类_时间规律_28日移动平均.png`、`品类_月度日均销量.png`、`品类_星期日均销量.png`：分别查看低频时间变化、月度/季节线索和星期规律。

## B. 单品销售量分布

单品层共有 **{int(item_summary['单品数'])} 个**实际出现销售记录的单品；三年销量中位数为 **{item_summary['销量中位数_kg']:.1f} kg**，四分位区间为 **[{item_summary['销量Q1_kg']:.1f}, {item_summary['销量Q3_kg']:.1f}] kg**。Top 10、Top 20 单品分别贡献总销量的 **{item_summary['Top10销量占比']:.1%}**、**{item_summary['Top20销量占比']:.1%}**。这说明集中程度的大小，但是否构成长尾结构应结合排名-累计贡献曲线判断，而非预设结论。

### 销量 Top 10

{markdown_table(item_stats, ['销量排名','单品名称','分类名称','总销量_kg','有销售日数','有销量日均销量_kg','CV_有销量日','累计销量占比'], 10)}

单品有销售日数中位数是 **{item_summary['销售天数中位数']:.0f} 天**，有销量日 CV 中位数是 **{item_summary['CV中位数']:.2f}**。这里的日均销量和 CV 均以“有销售记录日”为分母/样本，避免将上架前、退市后或未知状态的缺失机械视为零。

主要图：`figures/单品_总销量分布.png`、`单品_销量排名与累计贡献.png` 和 `单品_销售天数与波动分布.png`，分别用于检视分布形态、集中度及覆盖/波动差异。

## C. 品类之间的相互关系

品类关系同时在日尺度和周尺度计算 Pearson 与 Spearman 相关；周尺度减少了日内随机波动。绝对相关程度最高的一对是 **{cp['品类A']}—{cp['品类B']}**：日 Pearson={cp['日Pearson']:.2f}、日 Spearman={cp['日Spearman']:.2f}、周 Pearson={cp['周Pearson']:.2f}、周 Spearman={cp['周Spearman']:.2f}。

请通过四张热力图共同判断一对关系是否稳健：`品类_日尺度Pearson热力图.png`、`品类_日尺度Spearman热力图.png`、`品类_周尺度Pearson热力图.png`、`品类_周尺度Spearman热力图.png`。这些数值反映同步变化，不能说明一个品类导致另一个品类变化。

## D. 单品之间的相互关系：overlap 规则

全体 {int(item_corr['overlap_summary'].iloc[0]['单品对数']):,} 个单品对中，共同有销售记录日数的 Q1/中位数/Q3 分别为 **{item_corr['overlap_summary'].iloc[0]['Q1']:.0f}/{item_corr['overlap_summary'].iloc[0]['中位数']:.0f}/{item_corr['overlap_summary'].iloc[0]['Q3']:.0f} 天**；中位数只有 1 天，表明不应直接把 246 个单品补齐为 1095 天后计算矩阵。

主分析使用 **共同有销售记录不少于 30 天** 的门槛：它高于实际 Q3（{item_corr['overlap_summary'].iloc[0]['Q3']:.0f} 天），并提供约一个月的共同观测；有 {int(item_corr['overlap_summary'].iloc[0]['共同销售日≥30的单品对数']):,} 对满足条件。为便于敏感性检查，≥20 天和 ≥60 天的单品对数也在 `outputs/correlation/单品_overlap分布摘要.csv` 中保留。相关性在这类共同销售日期上计算，因而是条件性的销量同步指标。

### 共同销售日上的 Pearson 相关 Top 1

- 正相关：**{pos['单品名称_A']}**（{pos['分类名称_A']}）与 **{pos['单品名称_B']}**（{pos['分类名称_B']}），n={int(pos['共同有销售记录日数'])}，Pearson={pos['Pearson相关']:.2f}，Spearman={pos['Spearman相关']:.2f}。
- 负相关：**{neg['单品名称_A']}**（{neg['分类名称_A']}）与 **{neg['单品名称_B']}**（{neg['分类名称_B']}），n={int(neg['共同有销售记录日数'])}，Pearson={neg['Pearson相关']:.2f}，Spearman={neg['Spearman相关']:.2f}。

详细 Top 20 见 `outputs/correlation/单品_共同销售日不少于30天_正相关Top20.csv` 和 `…负相关Top20.csv`，散点图见 `figures/单品_正相关_散点图.png`、`单品_负相关_散点图.png`。正相关只支持“共同销售日上存在同步性”；负相关最多提示可能的替代性线索。二者都不足以证明互补、替代或因果。

## 【数据层】

本次使用现有问题一单品日表、品类日表及审计产出的活跃期表；未使用附件 3、附件 4。本轮未发现会推翻审计结论的新数据错误。十个无销售记录日继续保留为未观测日，不在本 EDA 中当作零。

## 【品类分布】

六个品类在规模、典型日销量和 CV 上存在明显差异；月度、星期和 28 日移动平均图给出时间规律的证据。当前可描述模式，尚不能由 EDA 判定驱动因素。

## 【单品分布】

单品销量、销售天数和有销量日波动差异显著；排名与累计贡献曲线用于判断集中程度。活跃期不同意味着不适合将所有未出现日期统一补零。

## 【品类关系】

日/周、Pearson/Spearman 四种口径并列呈现；只有在多种口径下方向与强度都一致时，才能较有把握地说需求变化具有同步或反向变化特征。

## 【单品关系】

先描述 overlap，再以 30 个共同销售日作为主门槛，并保存 20/60 天的敏感性计数。结果仅适用于共同销售日，后续如需讨论全活跃期关系，需要有可靠的“在售且零销量”状态数据。

## 【暂时结论】

数据已足以进入问题一的 EDA 审阅阶段。品类统计、时间图和多尺度相关矩阵可作为后续论文描述的实证基础；单品关系必须继续保留 overlap 条件和非因果表述。

## 【下一步】

审核本版的图形可读性、类别的月度/星期特征和相关对的业务合理性；若需要进入问题一第二轮，可针对稳健的品类相关对作时间序列复核，并按单品活跃期类型作分层描述。暂不进入预测或优化。
"""
    report_path.write_text(report, encoding="utf-8")


def create_second_round_report(category: dict, item: dict, corr: dict, item_corr: dict, final_tables: dict, report_path: Path) -> None:
    """Write the review-oriented report for the second EDA verification round."""
    cat_stats = category["stats"]
    item_stats = item["stats"]
    item_summary = item["summary"].iloc[0]
    focus = corr["focus"]
    sensitivity = item_corr["sensitivity"].iloc[0]
    robust = item_corr["robust"]
    seasonal = category["seasonal_stability"]
    structure = item_corr["structure_summary"].iloc[0]
    representatives = item_corr["representatives"]
    old_weeks, full_weeks = len(corr["weekly_old"]), len(corr["weekly"])
    robust_text = "未出现满足全部规则的单品对。" if robust.empty else (
        f"筛得 {len(robust)} 对；强度最高的是 **{robust.iloc[0]['单品名称_A']}—{robust.iloc[0]['单品名称_B']}** "
        f"（n={int(robust.iloc[0]['共同有销售记录日数'])}，Pearson={robust.iloc[0]['Pearson相关']:.2f}，Spearman={robust.iloc[0]['Spearman相关']:.2f}）。")
    report = f"""# 2023 年 CUMCM C 题：问题一 EDA（第三轮收口候选版）

> 本轮只验证问题一的统计规律与相关关系；不包含预测、补货、定价和问题二、问题三。运行 `problem1_eda.py` 可从问题一 CSV 重新生成本报告、所有表和图。

## 数据口径

- 品类日面板只含 **{category['observed_days']} 个有销售记录日**。三年日历中的 10 个全店无销售记录日既不填零，也不参与均值、周汇总或相关性。
- 在已观测日期，缺失的品类记录可记为该品类 0 销量；单品缺失记录不自动补零。
- 单品相关只在两件单品都有销售记录的共同日期计算，表示条件性的同步变化，不表示全在售期需求、更不表示因果。

### 统计口径与缺失处理

{markdown_table(final_tables['rules'], ['统计问题','有效样本','缺失处理','完整窗口要求'])}

核心原则是：**无销售记录日不等于销量 0；但缺少某一天，也不意味着附近其他有效日必须一起删除。只有统计量依赖固定完整时间窗口时，才要求窗口完整。** 因此仅周总销量和周尺度相关要求完整自然周。

## A. 描述性规律

### 六品类销售分布

{markdown_table(cat_stats, ['分类名称','总销量_kg','日均销量_kg','中位数_kg','标准差_kg','CV','Q1_kg','Q3_kg','偏度'])}

花叶类总销量最高（{cat_stats.iloc[0]['总销量_kg']:.1f} kg），茄类最低（{cat_stats.iloc[-1]['总销量_kg']:.1f} kg）；水生根茎类 CV 最高（{cat_stats['CV'].max():.2f}），花叶类最低（{cat_stats['CV'].min():.2f}）。这些仅为规模与波动的描述，不解释形成机制。

月度规律同时提供“合并月份平均”和 `品类_跨年度月度规律.png`。后者把 2020、2021、2022、2023 的年内月份分开绘制：2020 年只有 7—12 月、2023 年只有 1—6 月，因此跨年度重复性以完整的 2021、2022 年为主要依据，首尾两年只提供半年度旁证。28 日图已修正为**28 日历日窗口内的有效观测日平均**；长期趋势统计也改用距起始日期的实际日历天数，两项修正未改变品类规模、星期差异或相关关系的主结论。

### 2021/2022 月份曲线一致性

{markdown_table(seasonal, ['分类名称','2021_2022月均曲线Pearson','2021_2022月均曲线Spearman','2021_峰值月份','2022_峰值月份','2021_低谷月份','2022_低谷月份','判定'])}

在保守规则（两种曲线相关均≥0.50，且峰值、低谷月份均相差不超过 1 个月）下，没有品类满足“完整年度间有一定重复线索”。因此月份图可用于描述年度内变化，但本数据不能据此写成“已证明稳定季节性”。星期规律仍可描述为所有对应有效星期日的平均差异。

单品层共有 {int(item_summary['单品数'])} 个实际销售单品，Top 10/Top 20 分别贡献 {item_summary['Top10销量占比']:.1%}/{item_summary['Top20销量占比']:.1%} 总销量；销售天数中位数为 {item_summary['销售天数中位数']:.0f} 天。见 `单品_销量排名与累计贡献.png`、`单品_销售天数与波动分布.png`。

### 单品销量 Top 10

{markdown_table(item_stats, ['销量排名','单品名称','分类名称','总销量_kg','有销售日数','有销量日均销量_kg','CV_有销量日'], 10)}

## B. 控制时间规律后仍存在的品类关系

### 周尺度修正

第一版的 `resample("W-SUN").sum()` 共有 **{old_weeks} 周**，其中含有首尾不完整周和全店无销售记录日所在周。修正为仅保留周一至周日都有销售记录的完整自然周后，样本为 **{full_weeks} 周**，剔除 **{old_weeks-full_weeks} 周**。完整对照见 `outputs/correlation/品类_相关性_时间控制与周修正对照.csv`；周图应使用 `品类_周尺度完整自然周Pearson热力图.png` 与 `品类_周尺度完整自然周Spearman热力图.png`。

### 重点品类对：原始、修正周和残差相关

{markdown_table(focus, ['品类A','品类B','日Pearson','完整周Pearson','残差Pearson','残差Spearman','残差含年份Pearson','残差含年份Spearman'])}

主残差由星期虚拟变量、月份虚拟变量和实际日历日线性趋势的加性最小二乘模型得到；敏感性模型额外加入年份固定效应。三组重点对加入年份后方向和量级仍为正且相近，因而可称为“对该时间控制设定具有一定稳健性”。它没有控制价格、促销、供给等混杂因素，**不能作因果解释**。论文若保留品类关系，应优先呈现原始日尺度、完整自然周与残差热图均支持的方向。

## C. 仅作为线索的单品关系：真正的门槛敏感性分析

30,135 个单品对共同销售日数的 Q1/中位数/Q3 为 {item_corr['overlap_summary'].iloc[0]['Q1']:.0f}/{item_corr['overlap_summary'].iloc[0]['中位数']:.0f}/{item_corr['overlap_summary'].iloc[0]['Q3']:.0f} 天。分别重算 n≥30 与 n≥60 后，可计算的相关对数为 **{int(sensitivity['n≥30可计算单品对数']):,}** 和 **{int(sensitivity['n≥60可计算单品对数']):,}**；两种门槛下“最大绝对相关”Top20 的交集仅 **{int(sensitivity['n≥30绝对相关Top20与n≥60绝对相关Top20交集数']):,} 对**。

n≥30 中 Pearson/Spearman 异号对有 {int(sensitivity['n≥30 Pearson与Spearman异号对数']):,} 对，n≥60 中有 {int(sensitivity['n≥60 Pearson与Spearman异号对数']):,} 对。这些以及两指标绝对值相差大的单品对均不能当作稳健关系；详见 `单品_Pearson与Spearman差异Top50.csv`。

本报告定义“稳健相关对”为：共同销售日 n≥60、Pearson 与 Spearman 同号、且两者绝对值均≥0.40。{robust_text} 其中正相关 {int(structure['正相关对数'])} 对、负相关 {int(structure['负相关对数'])} 对；同品类 {int(structure['同品类对数'])} 对、跨品类 {int(structure['跨品类对数'])} 对。完整名单、品类组合结构和代表性样本分别见 `单品_稳健相关对_n不少于60.csv`、`单品_稳健相关对_品类组合汇总.csv`、`单品_稳健相关对_代表性样本.csv`。

### 代表性单品关系样本

{markdown_table(representatives, ['方向','单品名称_A','分类名称_A','单品名称_B','分类名称_B','共同有销售记录日数','Pearson相关','Spearman相关'], 5)}

代表样本要求共同销售日不低于稳健对的上四分位数，且两种相关差异不超过 0.10；它们更适合作为论文例子。负相关仅可称为反向同步或可能的替代性线索。

## 【数据层】

继续使用问题一单品日表、品类日表和审计活跃期表；附件 3、4 未参与。本轮未发现推翻既有审计的新数据问题。

## 【品类分布】

可使用总量、CV、箱线图、移动平均、星期图及经跨年度检查的月度图来描述规律；2020/2023 为半年度，跨年复现主要比较完整的 2021/2022；无法复现的月度差异不称为稳定季节性。

## 【单品分布】

单品销售规模、覆盖期和有销量日波动差异很大，不能把未出现日期统一填零。

## 【品类关系】

对完整自然周的修正和残差相关已提供。只有原始与控制时间规律后方向一致、且强度仍有解释价值的关系，才可以称为“控制时间规律后仍存在同步变化”。

## 【单品关系】

单品关系只代表共同销售日上的线索。Pearson/Spearman 异号或 n≥30、n≥60 排名不稳定的对不作为论文主要证据；稳健筛选也不证明替代、互补或因果。

## 【暂时结论】

最终收口表见 `outputs/final/结论_证据_限制.csv`。正文建议保留：品类总销量/箱线图、跨年度月度图、完整自然周相关热图、残差 Pearson 热图、单品累计贡献图、overlap 分布图，以及代表性稳健单品对表，共 5—8 张图。旧含不完整周相关、全量 Top20/Top100 和 Pearson-Spearman 差异表只放分析目录或附录，明确为修正对照，不作为论文结论。

建议正文按以下结论写作：品类分布写规模排序、相对波动差异、星期差异，并说明月份规律跨完整年度不稳定；单品分布写销量集中度、销售天数差异和不宜统一补零；品类关系写三组经完整周与残差验证后仍为正的同步线索；单品关系写 overlap 约束、筛选规则和少量代表性正相关，负相关仅写反向同步线索。

## 【下一步】

本轮到此暂停，等待审核；暂不进入预测或优化。
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Problem 1 reproducible EDA")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "问题1_建模数据")
    parser.add_argument("--item-daily", type=Path)
    parser.add_argument("--category-daily", type=Path)
    parser.add_argument("--active-stats", type=Path, default=ROOT / "analysis" / "problem1_audit_outputs" / "单品_活跃期统计.csv")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis" / "problem1")
    args = parser.parse_args()
    setup_style()
    item_path = args.item_daily or args.data_dir / "单品_日销售汇总.csv"
    category_path = args.category_daily or args.data_dir / "品类_日销售汇总.csv"
    analysis_dir = args.analysis_dir
    outputs, figures = analysis_dir / "outputs", analysis_dir / "figures"
    outputs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    item_daily = normalize_item_daily(read_csv(item_path))
    category_daily = normalize_category_daily(read_csv(category_path))
    active = read_csv(args.active_stats) if args.active_stats.exists() else None

    category = category_analysis(category_daily, outputs, figures)
    item = item_analysis(item_daily, active, outputs, figures)
    corr = category_correlations(category["panel"], outputs, figures)
    item_corr = item_overlap_and_correlations(item_daily, active, outputs, figures)
    final_tables = write_final_candidate_tables(category, corr, item_corr, outputs)
    create_second_round_report(category, item, corr, item_corr, final_tables, analysis_dir / "problem1_eda_report.md")
    print(f"Wrote EDA report, tables and figures under: {analysis_dir}")


if __name__ == "__main__":
    main()


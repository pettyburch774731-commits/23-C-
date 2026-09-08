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

    weekdays = panel.copy()
    weekdays["星期"] = weekdays.index.dayofweek
    weekday_avg = weekdays.groupby("星期")[names].mean().reindex(range(7)).reset_index()
    weekday_avg["星期名称"] = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    write_csv(weekday_avg, outputs / "category" / "品类_星期日均销量.csv")

    trend = pd.DataFrame({
        "分类名称": names,
        "Spearman_日期趋势": [panel[name].corr(pd.Series(np.arange(len(panel)), index=panel.index), method="spearman") for name in names],
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

    rolling = panel.rolling(28, min_periods=14).mean()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for name in names:
        ax.plot(rolling.index, rolling[name], linewidth=1.2, label=name)
    ax.set(title="品类日销量的 28 日移动平均", xlabel="日期", ylabel="日销量（kg）")
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

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(7)
    width = 0.12
    for i, name in enumerate(names):
        ax.bar(x + (i - 2.5) * width, weekday_avg[name], width=width, label=name)
    ax.set(title="不同星期的品类平均日销量", xlabel="星期", ylabel="平均日销量（kg）", xticks=x, xticklabels=weekday_avg["星期名称"])
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(figures / "品类_星期日均销量.png")
    plt.close(fig)

    return {"panel": panel, "stats": stats, "monthly": month_avg, "weekday": weekday_avg, "trend": trend, "observed_days": len(observed_dates)}


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


def category_correlations(panel: pd.DataFrame, outputs: Path, figures: Path) -> dict:
    daily_p = panel.corr(method="pearson")
    daily_s = panel.corr(method="spearman")
    weekly = panel.resample("W-SUN").sum()
    weekly_p = weekly.corr(method="pearson")
    weekly_s = weekly.corr(method="spearman")
    for name, table in [("品类_日尺度_Pearson相关矩阵.csv", daily_p), ("品类_日尺度_Spearman相关矩阵.csv", daily_s),
                        ("品类_周尺度_Pearson相关矩阵.csv", weekly_p), ("品类_周尺度_Spearman相关矩阵.csv", weekly_s)]:
        write_csv(table.reset_index().rename(columns={"category_name": "分类名称"}), outputs / "correlation" / name)
    heatmap(daily_p, "品类日销量 Pearson 相关", figures / "品类_日尺度Pearson热力图.png")
    heatmap(daily_s, "品类日销量 Spearman 相关", figures / "品类_日尺度Spearman热力图.png")
    heatmap(weekly_p, "品类周销量 Pearson 相关", figures / "品类_周尺度Pearson热力图.png")
    heatmap(weekly_s, "品类周销量 Spearman 相关", figures / "品类_周尺度Spearman热力图.png")

    pairs = []
    for a, b in itertools.combinations(panel.columns, 2):
        pairs.append({"品类A": a, "品类B": b, "日Pearson": daily_p.loc[a, b], "日Spearman": daily_s.loc[a, b],
                      "周Pearson": weekly_p.loc[a, b], "周Spearman": weekly_s.loc[a, b]})
    pair_table = pd.DataFrame(pairs)
    pair_table["最大绝对相关"] = pair_table[["日Pearson", "日Spearman", "周Pearson", "周Spearman"]].abs().max(axis=1)
    pair_table = pair_table.sort_values("最大绝对相关", ascending=False)
    write_csv(pair_table, outputs / "correlation" / "品类_相关性对照表.csv")
    return {"daily_p": daily_p, "daily_s": daily_s, "weekly_p": weekly_p, "weekly_s": weekly_s, "pairs": pair_table, "weekly": weekly}


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
    pos = correlations.sort_values("Pearson相关", ascending=False).head(20)
    neg = correlations.sort_values("Pearson相关", ascending=True).head(20)
    write_csv(pos, outputs / "correlation" / "单品_共同销售日不少于30天_正相关Top20.csv")
    write_csv(neg, outputs / "correlation" / "单品_共同销售日不少于30天_负相关Top20.csv")
    write_csv(correlations.sort_values("最大绝对相关", ascending=False).head(100), outputs / "correlation" / "单品_相关性绝对值Top100.csv")

    # Plot the largest positive and negative Pearson examples for inspection.
    examples = [("正", pos.iloc[0]), ("负", neg.iloc[0])]
    for label, row in examples:
        joined = pd.concat([values[row["单品编码_A"]], values[row["单品编码_B"]]], axis=1, join="inner")
        joined.columns = [row["单品名称_A"], row["单品名称_B"]]
        fig, ax = plt.subplots(figsize=(5.5, 4.7))
        sns.regplot(data=joined, x=row["单品名称_A"], y=row["单品名称_B"], scatter_kws={"s": 15, "alpha": .55}, line_kws={"color": "#E45756"}, ax=ax)
        ax.set(title=f"单品共同销售日数量关系：{label}相关\nPearson={row['Pearson相关']:.2f}, n={int(row['共同有销售记录日数'])}")
        fig.savefig(figures / f"单品_{label}相关_散点图.png")
        plt.close(fig)

    return {"overlap_summary": overlap_summary, "distribution": distribution, "threshold": threshold, "eligible_pairs": len(eligible), "correlations": correlations, "positive": pos, "negative": neg}


def fmt(value: float, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    selected = df[columns].head(n) if n else df[columns]
    return selected.to_markdown(index=False, floatfmt=".2f")


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
    create_report(category, item, corr, item_corr, analysis_dir / "problem1_eda_report.md")
    print(f"Wrote EDA report, tables and figures under: {analysis_dir}")


if __name__ == "__main__":
    main()


"""Reproducible analyses for 2023 CUMCM C problems 2--4."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

RNG = np.random.default_rng(202309)
TARGET_DATE = pd.Timestamp("2023-07-01")
FUTURE_DATES = pd.date_range(TARGET_DATE, periods=7, freq="D")


def read_inputs(input_dir: Path):
    product = pd.read_excel(input_dir / "附件1.xlsx")
    sales = pd.read_excel(input_dir / "附件2.xlsx")
    cost = pd.read_excel(input_dir / "附件3.xlsx")
    loss = pd.read_excel(input_dir / "附件4.xlsx")
    product.columns = ["item_id", "item_name", "category_id", "category"]
    sales.columns = ["sale_date", "scan_time", "item_id", "quantity", "unit_price", "sale_type", "discount"]
    cost.columns = ["date", "item_id", "wholesale_price"]
    loss.columns = ["category_id", "category", "loss_pct"]
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])
    cost["date"] = pd.to_datetime(cost["date"])
    for x in (product, sales, cost, loss):
        for c in x.columns:
            if c.endswith("_id") or c == "item_id":
                x[c] = pd.to_numeric(x[c], errors="coerce").astype("Int64")
    loss["loss_rate"] = pd.to_numeric(loss["loss_pct"], errors="coerce") / 100
    return product, sales, cost, loss


def forward_fill_cost(cost: pd.DataFrame, product: pd.DataFrame):
    """Only use previous observed price when a unit cost is missing."""
    full_days = pd.date_range(cost.date.min(), TARGET_DATE - pd.Timedelta(days=1), freq="D")
    ids = product.item_id.dropna().astype(int).unique()
    grid = pd.MultiIndex.from_product([ids, full_days], names=["item_id", "date"]).to_frame(index=False)
    x = grid.merge(cost, how="left", on=["item_id", "date"]).sort_values(["item_id", "date"])
    x["cost_observed"] = x.wholesale_price.notna()
    x["wholesale_price"] = x.groupby("item_id")["wholesale_price"].ffill()
    x["cost_imputed"] = ~x.cost_observed & x.wholesale_price.notna()
    return x


def build_daily(product, sales, cost, loss):
    normal = sales.loc[sales.sale_type.eq("销售")].copy()
    normal = normal.merge(product, on="item_id", how="left", validate="many_to_one")
    normal["revenue"] = normal.quantity * normal.unit_price
    filled_cost = forward_fill_cost(cost, product)
    normal = normal.merge(filled_cost, left_on=["item_id", "sale_date"], right_on=["item_id", "date"], how="left")
    normal["cost_amount"] = normal.quantity * normal.wholesale_price
    g = normal.groupby(["sale_date", "category_id", "category"], as_index=False)
    daily = g.agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"),
                  cost_amount=("cost_amount", "sum"), cost_weight=("quantity", "sum"),
                  transaction_count=("quantity", "size"), cost_imputed_rows=("cost_imputed", "sum"))
    daily["price"] = daily.revenue / daily.quantity
    daily["cost"] = daily.cost_amount / daily.cost_weight
    daily = daily.merge(loss[["category_id", "loss_rate"]], on="category_id", how="left")
    daily["markup"] = (daily.price - daily.cost) / daily.cost
    daily["weekday"] = daily.sale_date.dt.weekday
    daily["month"] = daily.sale_date.dt.month
    daily["trend"] = (daily.sale_date - daily.sale_date.min()).dt.days / 1000
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(subset=["quantity", "price", "cost", "markup"])
    return normal, daily, filled_cost


def design(df, kind):
    variable = np.log(df.price.clip(lower=0.05)).to_numpy() if kind == "log_price" else df.markup.to_numpy()
    cols = [np.ones(len(df)), variable]
    if kind == "quad_markup":
        cols.append(variable ** 2)
    for d in range(1, 7):
        cols.append((df.weekday.to_numpy() == d).astype(float))
    for m in range(2, 13):
        cols.append((df.month.to_numpy() == m).astype(float))
    cols.append(df.trend.to_numpy())
    return np.column_stack(cols)


@dataclass
class DemandFit:
    kind: str
    beta: np.ndarray
    residuals: np.ndarray
    mae: float
    elasticity: float
    support_low: float
    support_high: float

    def predict(self, df):
        raw = design(df, self.kind) @ self.beta
        return np.exp(raw) if self.kind == "log_price" else np.maximum(raw, 0.0)


def fit_demand(df, kind):
    x = design(df, kind)
    y = np.log(df.quantity.clip(lower=0.05)).to_numpy() if kind == "log_price" else df.quantity.to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    raw = x @ beta
    pred = np.exp(raw) if kind == "log_price" else np.maximum(raw, 0.0)
    residuals = df.quantity.to_numpy() - pred
    if kind == "log_price":
        elasticity = float(beta[1])
    elif kind == "quad_markup":
        elasticity = float((beta[1] + 2 * beta[2] * df.markup.median()) * (1 + df.markup.median()) / max(df.quantity.mean(), .1))
    else:
        elasticity = float(beta[1] * (1 + df.markup.median()) / max(df.quantity.mean(), .1))
    return DemandFit(kind, beta, residuals, np.nan, elasticity, float(df.markup.quantile(.05)), float(df.markup.quantile(.95)))


def blocked_validation(df, kind, horizon=7):
    dates = np.sort(df.sale_date.unique())
    starts = range(max(150, len(dates) - 182), len(dates) - horizon, 14)
    errors = []
    for s in starts:
        tr_dates, te_dates = dates[:s], dates[s:s + horizon]
        train, test = df[df.sale_date.isin(tr_dates)], df[df.sale_date.isin(te_dates)]
        if len(train) < 120 or len(test) == 0:
            continue
        errors.extend(np.abs(test.quantity.to_numpy() - fit_demand(train, kind).predict(test)))
    return float(np.mean(errors)) if errors else np.inf


def select_demand_models(daily):
    rows, models = [], {}
    for category, df in daily.groupby("category"):
        df = df.sort_values("sale_date").copy()
        candidates = ["linear_markup", "quad_markup", "log_price"]
        scores = {k: blocked_validation(df, k) for k in candidates}
        chosen = min(scores, key=scores.get)
        model = fit_demand(df, chosen)
        model.mae = scores[chosen]
        models[category] = model
        rows.append({"category": category, "selected_model": chosen, "blocked_7d_mae": model.mae,
                     "response_or_elasticity": model.elasticity, "markup_p05": model.support_low,
                     "markup_p95": model.support_high, **{f"mae_{k}": v for k, v in scores.items()}})
    return models, pd.DataFrame(rows)


def pred_cost(history, method):
    h = pd.Series(history).dropna().to_numpy(dtype=float)
    if len(h) == 0:
        return np.nan
    if method == "last":
        return h[-1]
    if method.startswith("mean"):
        return float(np.mean(h[-int(method[4:]):]))
    return float(pd.Series(h).ewm(alpha=.3, adjust=False).mean().iloc[-1])


def choose_cost_model(series):
    y = pd.Series(series).dropna()
    methods = ["last", "mean7", "mean14", "mean28", "ewma"]
    scores = {}
    for method in methods:
        errors = []
        for o in range(max(35, len(y) - 140), len(y) - 7, 7):
            forecast = pred_cost(y.iloc[:o], method)
            errors.extend(np.abs(y.iloc[o:o + 7].to_numpy() - forecast))
        scores[method] = float(np.mean(errors)) if errors else np.inf
    return min(scores, key=scores.get), scores


def cost_forecasts(daily):
    forecasts, rows = {}, []
    for category, df in daily.groupby("category"):
        series = df.sort_values("sale_date").set_index("sale_date").cost
        best, scores = choose_cost_model(series)
        history = series.copy()
        values = []
        for date in FUTURE_DATES:
            forecast = pred_cost(history, best)
            values.append(forecast)
            history.loc[date] = forecast
        forecasts[category] = values
        rows.append({"category": category, "selected_cost_method": best, "blocked_7d_mae": scores[best],
                     **{f"mae_{k}": v for k, v in scores.items()}})
    table = pd.DataFrame({"date": FUTURE_DATES})
    for category, values in forecasts.items():
        table[category] = values
    return table.melt(id_vars="date", var_name="category", value_name="forecast_cost"), pd.DataFrame(rows)


def future_features(date, cost, markup, anchor):
    return pd.DataFrame([{"sale_date": date, "cost": cost, "markup": markup, "price": cost * (1 + markup),
                          "weekday": date.weekday(), "month": date.month,
                          "trend": (date - anchor).days / 1000}])


def optimize_categories(daily, models, cost_fc):
    output, baseline = [], []
    anchor = daily.sale_date.min()
    for _, cost_row in cost_fc.iterrows():
        category, date, cost = cost_row.category, pd.Timestamp(cost_row.date), float(cost_row.forecast_cost)
        history = daily[daily.category.eq(category)].sort_values("sale_date")
        model = models[category]
        loss = float(history.loss_rate.iloc[0])
        choices = []
        for markup in np.linspace(model.support_low, model.support_high, 61):
            feature = future_features(date, cost, markup, anchor)
            demand = float(model.predict(feature)[0])
            samples = np.maximum(0, demand + RNG.choice(model.residuals, size=1200, replace=True))
            price = float(feature.price.iloc[0])
            effective_cost = cost / (1 - loss)
            critical = np.clip(1 - effective_cost / price, .01, .99)
            sellable_stock = float(np.quantile(samples, critical))
            replenishment = sellable_stock / (1 - loss)
            expected_sales = float(np.minimum(samples, sellable_stock).mean())
            profit = price * expected_sales - cost * replenishment
            choices.append((profit, markup, price, demand, replenishment, expected_sales))
        profit, markup, price, demand, replenishment, expected_sales = max(choices)
        output.append({"date": date, "category": category, "forecast_cost": cost, "loss_rate": loss,
                       "recommended_markup": markup, "recommended_price": price,
                       "recommended_replenishment_kg": replenishment, "expected_demand_kg": demand,
                       "expected_sales_kg": expected_sales, "expected_revenue": price * expected_sales,
                       "expected_cost": cost * replenishment, "expected_profit": profit,
                       "markup_support_p05": model.support_low, "markup_support_p95": model.support_high,
                       "price_in_support": model.support_low <= markup <= model.support_high})
        base_markup = float(history.markup.mean())
        base_demand = float(history.tail(7).quantity.mean())
        base_price = cost * (1 + np.clip(base_markup, model.support_low, model.support_high))
        base_replenishment = base_demand / (1 - loss)
        baseline.append({"date": date, "category": category, "baseline_markup": base_markup,
                         "baseline_price": base_price, "baseline_replenishment_kg": base_replenishment,
                         "baseline_expected_profit": base_price * base_demand - cost * base_replenishment})
    return pd.DataFrame(output), pd.DataFrame(baseline)


def category_sensitivity(plan, models, daily):
    scenarios = [("elasticity_-20pct", .8, 1, 0), ("elasticity_+20pct", 1.2, 1, 0),
                 ("cost_+10pct", 1, 1.1, 0), ("loss_+5pp", 1, 1, .05)]
    rows = []
    for scenario, e_scale, c_scale, l_add in scenarios:
        profits = []
        for _, row in plan.iterrows():
            history = daily[daily.category.eq(row.category)]
            model = models[row.category]
            adjusted_e = model.elasticity * e_scale
            ref_price = history.price.median()
            demand_ratio = (row.recommended_price / ref_price) ** (adjusted_e - model.elasticity)
            loss = min(.45, row.loss_rate + l_add)
            cost = row.forecast_cost * c_scale
            # Start from the bootstrap expected sales in the base plan, rather
            # than replacing uncertainty with deterministic demand.
            sales = min(row.expected_sales_kg * demand_ratio, (1 - loss) * row.recommended_replenishment_kg)
            profits.append(row.recommended_price * sales - cost * row.recommended_replenishment_kg)
        rows.append({"scenario": scenario, "expected_profit": sum(profits),
                     "change_vs_base": sum(profits) / plan.expected_profit.sum() - 1})
    return pd.DataFrame(rows)


def item_cost_forecast(cost_filled, item_id):
    series = cost_filled[cost_filled.item_id.eq(item_id)].sort_values("date").wholesale_price.dropna()
    if len(series) >= 7:
        method, _ = choose_cost_model(series)
        return float(pred_cost(series, method)), method, False
    return (float(series.iloc[-1]) if len(series) else np.nan), "last_valid_fallback", True


def item_demand_and_options(normal, cost_filled, loss, category_models):
    week = normal[(normal.sale_date >= TARGET_DATE - pd.Timedelta(days=7)) & (normal.sale_date < TARGET_DATE)]
    candidate_week = week.groupby(["item_id", "item_name", "category_id", "category"], as_index=False).quantity.sum()
    candidate_week = candidate_week[candidate_week.quantity > 0].copy()
    rows, options = [], []
    for _, item in candidate_week.iterrows():
        iid, category = int(item.item_id), item.category
        raw_history = normal[(normal.item_id.eq(iid)) & (normal.sale_date < TARGET_DATE)].copy()
        # Aggregate transaction lines to item-day observations before taking
        # recent 7/28-day averages; tail(7) transaction lines is not a day demand.
        history = (raw_history.groupby("sale_date", as_index=False)
                   .agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"),
                        cost_amount=("cost_amount", "sum"))
                   .sort_values("sale_date"))
        history["unit_price"] = history.revenue / history.quantity
        history["wholesale_price"] = history.cost_amount / history.quantity
        q7, q28 = history.tail(7).quantity.mean(), history.tail(28).quantity.mean()
        same_day = history[history.sale_date.dt.weekday.eq(TARGET_DATE.weekday())].tail(8).quantity.mean()
        base = max(float(np.nanmean([q7, q28, same_day])), .1)
        priced = history.dropna(subset=["wholesale_price"]).copy()
        priced["markup"] = (priced.unit_price - priced.wholesale_price) / priced.wholesale_price
        if priced.empty:
            continue
        price_ref = float(priced.unit_price.tail(28).median())
        category_e = float(category_models[category].elasticity)
        elasticity, source = category_e, "category_shrinkage"
        if len(priced) >= 90 and priced.unit_price.nunique() >= 10 and (priced.quantity > 0).all():
            own = np.polyfit(np.log(priced.unit_price.to_numpy()), np.log(priced.quantity.to_numpy()), 1)[0]
            elasticity = len(priced) / (len(priced) + 90) * own + 90 / (len(priced) + 90) * category_e
            source = "item_category_shrinkage"
        elasticity = float(np.clip(elasticity, -3, 1))
        cost, cost_method, fallback = item_cost_forecast(cost_filled, iid)
        if not np.isfinite(cost):
            continue
        loss_rate = float(loss.loc[loss.category_id.eq(item.category_id), "loss_rate"].iloc[0])
        marks = priced.markup.replace([np.inf, -np.inf], np.nan).dropna()
        low, high = (marks.quantile(.10), marks.quantile(.90)) if len(marks) >= 10 else (.10, .60)
        low, high = max(-.05, float(low)), max(float(low) + .05, float(high))
        rows.append({**item.to_dict(), "base_demand_kg": base, "category_elasticity": category_e,
                     "item_elasticity": elasticity, "elasticity_source": source, "forecast_cost": cost,
                     "cost_method": cost_method, "cost_fallback": fallback, "loss_rate": loss_rate,
                     "markup_low": low, "markup_high": high})
        for option, markup in enumerate(np.linspace(low, high, 5)):
            price = cost * (1 + markup)
            demand = float(np.clip(base * (price / price_ref) ** elasticity, .2 * base, 3 * base))
            # A selected perishable item may cover at most two forecast sales days.
            # This is a finite, demand-based upper inventory rule, not a free stock variable.
            stock = max(2.0 * demand, (1 - loss_rate) * 2.5)
            replenishment = max(stock / (1 - loss_rate), 2.5)
            expected_sales = min(demand, stock)
            options.append({"item_id": iid, "option": option, "category": category,
                            "category_id": int(item.category_id), "item_name": item.item_name,
                            "price": price, "markup": markup, "cost": cost, "loss_rate": loss_rate,
                            "demand": demand, "effective_inventory": (1-loss_rate)*replenishment,
                            "replenishment": replenishment, "expected_sales": expected_sales,
                            "profit": price*expected_sales-cost*replenishment})
    return pd.DataFrame(rows), pd.DataFrame(options), candidate_week


def solve_item_milp(options, category_targets, loss_override=0, demand_scale=1, cost_scale=1):
    opt = options.copy()
    opt["scenario_demand"] = opt.demand * demand_scale
    opt["scenario_effective_inventory"] = opt.replenishment * (1 - (opt.loss_rate + loss_override))
    opt["scenario_profit"] = opt.price*np.minimum(opt.scenario_demand, opt.scenario_effective_inventory) - opt.cost*cost_scale*opt.replenishment
    n = len(opt)
    coefficients, lower, upper = [np.ones(n)], [27], [33]
    for _, idx in opt.groupby("item_id").groups.items():
        a = np.zeros(n); a[list(idx)] = 1
        coefficients.append(a); lower.append(0); upper.append(1)
    for category, target in category_targets.items():
        mask = opt.category.eq(category).to_numpy()
        a = np.zeros(n); a[mask] = opt.loc[mask, "scenario_effective_inventory"]
        coefficients.append(a); lower.append(target*demand_scale); upper.append(np.inf)
    result = milp(c=-opt.scenario_profit.to_numpy(), integrality=np.ones(n),
                  bounds=Bounds(np.zeros(n), np.ones(n)),
                  constraints=LinearConstraint(np.vstack(coefficients), np.array(lower), np.array(upper)),
                  options={"time_limit": 120})
    if not result.success:
        raise RuntimeError("MILP failed: " + result.message)
    return opt.loc[result.x > .5].copy(), result


def item_sensitivity(options, targets, base_plan):
    rows, base_items = [], set(base_plan.item_id)
    cases = [("demand_-10pct", .9, 1, 0), ("demand_+10pct", 1.1, 1, 0),
             ("cost_+10pct", 1, 1.1, 0), ("loss_+5pp", 1, 1, .05)]
    for name, d_scale, c_scale, loss_add in cases:
        solution, _ = solve_item_milp(options, targets, loss_add, d_scale, c_scale)
        selected = set(solution.item_id)
        rows.append({"scenario": name, "selected_count": len(selected),
                     "expected_profit": solution.scenario_profit.sum(),
                     "common_items": len(selected & base_items),
                     "jaccard_vs_base": len(selected & base_items) / len(selected | base_items)})
    return pd.DataFrame(rows)


def write_markdown_reports(out, plan, baseline, demand_models, cost_models, cat_sens, candidates, item_plan, checks, item_sens):
    p2, p3, paper = out/"analysis/problem2", out/"analysis/problem3", out/"paper"
    for folder in (p2/"tables", p2/"figures", p3/"tables", p3/"figures", paper):
        folder.mkdir(parents=True, exist_ok=True)
    plan.to_csv(p2/"tables/七日品类补货定价策略.csv", index=False, encoding="utf-8-sig")
    baseline.to_csv(p2/"tables/朴素基线策略.csv", index=False, encoding="utf-8-sig")
    demand_models.to_csv(p2/"tables/需求模型时间验证.csv", index=False, encoding="utf-8-sig")
    cost_models.to_csv(p2/"tables/成本模型时间验证.csv", index=False, encoding="utf-8-sig")
    cat_sens.to_csv(p2/"tables/敏感性分析.csv", index=False, encoding="utf-8-sig")
    (plan.groupby("date", as_index=False)
     .agg(expected_revenue=("expected_revenue", "sum"), expected_cost=("expected_cost", "sum"),
          expected_profit=("expected_profit", "sum"))
     .to_csv(p2/"tables/每日利润汇总.csv", index=False, encoding="utf-8-sig"))
    candidates.to_csv(p3/"tables/候选单品与需求成本估计.csv", index=False, encoding="utf-8-sig")
    item_plan.to_csv(p3/"tables/2023-07-01单品补货定价策略.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(p3/"tables/约束检查.csv", index=False, encoding="utf-8-sig")
    item_sens.to_csv(p3/"tables/敏感性分析.csv", index=False, encoding="utf-8-sig")
    (item_plan.groupby("category", as_index=False)
     .agg(selected_items=("item_id", "nunique"), replenishment_kg=("replenishment", "sum"),
          effective_inventory_kg=("effective_inventory", "sum"), expected_profit=("expected_profit", "sum"))
     .to_csv(p3/"tables/品类汇总.csv", index=False, encoding="utf-8-sig"))
    total, base_total = plan.expected_profit.sum(), baseline.baseline_expected_profit.sum()
    (paper/"problem2_draft.md").write_text(f"""# 问题二：品类级补货与定价

## 1. 建模口径与目标

以正常销售记录构造品类—日面板。日销量、销售额、售价和批发成本均为销量加权口径；退货不进入需求建模。若单品当日批发价缺失，只使用该单品此前已观测价格向前填补，并保留插补标记；因此没有把未来成本信息带入 2023 年 7 月的决策。全店无销售记录日不补零。

设品类 $c$ 在日期 $t$ 的销量、销量加权售价和销量加权批发成本分别为 $Q_{{c,t}},P_{{c,t}},C_{{c,t}}$，成本加成率为

$$m_{{c,t}}=\frac{{P_{{c,t}}-C_{{c,t}}}}{{C_{{c,t}}}}.$$

需求候选模型包含线性加成率、二次加成率和对数价格模型；均控制星期、月份和线性时间趋势。例如，对数价格模型为

$$\log Q_{{c,t}}=\alpha_c+\beta_c\log P_{{c,t}}+\gamma_{{\rm weekday}}+\delta_{{\rm month}}+\eta t+\varepsilon_{{c,t}}.$$

以滚动的连续 7 日测试块比较 MAE，选择每类误差最低的模型。该关系是控制共同时间规律后的历史预测关系，不能解释为严格的因果价格弹性。

## 2. 成本预测、定价和补货

每个品类在最近值、近 7/14/28 日均值和指数平滑五种成本预测方法中，仍以滚动 7 日 MAE 选择。对 2023-07-01 至 07-07 的成本递推预测。候选加成率仅在各类别历史 5%--95% 分位的支持区间内搜索，避免无依据外推。

附件 4 给出品类损耗率 $L_c$。将可销售库存记为 $y=(1-L_c)R$，则有效成本为 $C^{{eff}}=C/(1-L_c)$。在零残值近似下，利用需求模型残差 bootstrap 形成需求样本，并以报童临界分位数

$$F_D(y^*)=1-\frac{{C^{{eff}}}}{{P}}$$

确定库存，再换算采购量 $R^*=y^*/(1-L_c)$。对每一日和品类，直接在候选价格与对应库存上最大化

$$E[\Pi]=E\left[P\min\{{D,(1-L_c)R\}}-CR\right].$$

## 3. 结果与验证

表 `analysis/problem2/tables/需求模型时间验证.csv` 和 `成本模型时间验证.csv` 给出全部候选方法的滚动 MAE 及最终选择；`七日品类补货定价策略.csv` 给出 42 组日期—品类策略。七日预计利润为 **{total:.2f} 元**；以历史平均加成率、最近 7 日平均销量补货的朴素基线为 **{base_total:.2f} 元**，同一预测框架下相差 **{total-base_total:.2f} 元**。每日收益汇总和价格路径见对应表与图。

价格均在历史支持区间内，代码对该条件、时间顺序和约束口径进行自动检查。敏感性结果保存在 `敏感性分析.csv`：成本上升会明显压缩利润，价格关系强度变化亦会改变结果；损耗率上调情景若未改变库存约束，数值影响可能较小，不能据此推断损耗不重要。结论仍受历史价格内生性、固定损耗率和实际库存未知等限制。
""", encoding="utf-8")
    (paper/"problem3_draft.md").write_text("""# 问题三：2023 年 7 月 1 日单品级补货与定价

## 1. 候选集与预测

候选集只包括 2023-06-24 至 2023-06-30 出现正常销售记录的单品，以该周出现正常销售作为可售状态的代理，而非将全部历史单品视作 7 月 1 日可售。单品基础需求结合最近 7 日、最近 28 日和历史同星期几的销量。样本充分且价格变化足够的单品估计自身对数价格响应，并向问题二对应品类的响应收缩；样本不足的单品直接采用品类收缩响应。该策略避免为稀疏单品分别拟合不稳定的时间序列。

候选单品的批发成本通过历史滚动回测选取的简单成本方法预测；若序列不足，回退到最近有效批发价，并在候选表中标记。损耗率直接沿用其所属品类的附件 4 值，令 $C_i^{eff}=C_i/(1-L_i)$。每个候选单品在自身历史合理加成率范围内生成 5 个价格档，并预计算需求、补货量和利润。

## 2. 整数规划模型

令 $z_{i,k}\in\{{0,1}}$ 表示选择单品 $i$ 的第 $k$ 个价格档。模型最大化所有单品—价格档的预计算预期利润。主要约束为

$$27\leq\sum_{{i,k}}z_{{i,k}}\leq33,\qquad \sum_k z_{{i,k}}\leq1,$$

并对每个选中单品强制 $R_i\geq2.5$ kg。库存上限采用至多覆盖两日预测需求的有限、需求驱动规则，避免不受约束的大库存。对每个品类 $c$，有效库存满足问题二 7 月 1 日需求目标：

$$\sum_{{i\in c}}(1-L_i)R_i\geq D^{{target}}_{{c,2023\text{{-}}07\text{{-}}01}}.$$

## 3. 结果、约束检查与限制

完整策略见 `analysis/problem3/tables/2023-07-01单品补货定价策略.csv`；候选单品需求、成本、收缩来源和回退标记见 `候选单品与需求成本估计.csv`。`约束检查.csv` 是程序自动生成的检查表，逐项验证 27--33 个品种、每项不少于 2.5 kg、MILP 求解成功和六个品类有效库存目标。

需求 ±10%、成本 +10% 和损耗率 +5 个百分点的重优化结果记录在 `敏感性分析.csv`，同时报告与基准入选集的 Jaccard 相似度。该题的强制品类需求、27--33 个品种限制和未知真实库存使收益对假设敏感，尤其类别层面出现负的预测利润时，应理解为约束张力的诊断，而非可直接执行的保证收益方案。实际订货、到货、陈列容量和库存状态是后续落地前必须补齐的信息。
""", encoding="utf-8")
    (paper/"problem4_draft.md").write_text("""# 问题四：建议新增采集的数据

| 优先级 | 数据 | 当前缺失造成的问题 | 收集后改善的环节 |
|---:|---|---|---|
| 1 | 每日开店库存、到货、上架时刻、缺货时段、日终库存 | 无法区分零需求、未上架与缺货截断 | 问题一单品缺失解释；问题二、三真实需求和库存约束 |
| 2 | 实际订货量、到货量、到货时间、供应商缺货 | 历史销售无法验证订货是否可执行 | 问题二、三补货训练和供给可行性 |
| 3 | 批次损耗、报损时间、剩余库存、折价处理量 | 只有长期平均损耗率，无法描述动态损耗 | 问题二、三的有效成本和订货量 |
| 4 | 供应商报价、起订量、能力、提前期、产地品质 | 成本预测和订货约束过于简化 | 问题二成本预测；问题三采购约束 |
| 5 | 促销、会员价、优惠券、临期折价、陈列促销 | 价格与销量关系混入促销效应 | 问题二价格关系；问题三单品定价 |
| 6 | 货架面积、位置、最大容量、最小陈列量 | 单品组合没有真实空间约束 | 问题三选品与陈列可行性 |
| 7 | 法定节假日、调休、天气、门店客流 | 星期与短期需求的解释受限 | 问题一时间规律；问题二、三短期预测 |
| 8 | 顾客购物篮 | 销量相关不能区分共同购买与替代 | 问题一单品关系；问题三组合决策 |
| 9 | 竞争门店和线上价格 | 自身价格关系混入市场竞争 | 问题二、三价格敏感性 |

这些是建议采集的数据，不表示现有分析已经证明其造成销量变化。
""", encoding="utf-8")
    html = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>2023 C题问题二至四报告</title><style>body{font:16px/1.7 Arial,'Microsoft YaHei';max-width:1200px;margin:32px auto;color:#1f2937}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #ccd6e0;padding:7px}th{background:#edf4fa}h2{border-left:5px solid #185a9d;padding-left:10px}</style>"
    plan_html = plan.copy()
    plan_html["date"] = pd.to_datetime(plan_html["date"]).dt.strftime("%Y-%m-%d")
    item_html = item_plan.copy()
    html += (f"<h1>2023 C题：问题二至四分析报告</h1><p>只使用 2023-06-30 及以前数据。"
             f"问题二七日品类策略的预计利润为 {total:.2f} 元；朴素基线为 {base_total:.2f} 元。</p>"
             "<h2>问题二：方法与七日策略</h2><p>需求模型控制星期、月份和时间趋势，按滚动 7 日 MAE 选择；"
             "成本模型按同一时间顺序回测选择。价格仅在历史加成率 5%--95% 区间搜索；"
             "损耗通过有效库存和报童式残差 bootstrap 进入补货决策。</p>" + plan_html.round(3).to_html(index=False))
    html += (f"<h2>问题三：7月1日单品策略</h2><p>候选 {len(candidates)} 个，选中 {len(item_plan)} 个，"
             f"预计利润 {item_plan.expected_profit.sum():.2f} 元。候选仅来自 6 月 24--30 日有正常销售的单品；"
             "MILP 自动检查选品数量、2.5 kg 下限及六品类的有效库存目标。</p>" + item_html.round(3).to_html(index=False))
    html += "<h2>问题四：建议采集的数据</h2><p>优先补齐实际库存/在售、实际订货到货和动态损耗数据；其余优先级与对问题一至三的作用见 problem4_draft.md。</p><h2>阅读限制</h2><p>这些结果是基于历史观测数据的预测优化，不能把价格关系或品类/单品同步关系解释为因果。实际库存、供应能力和动态损耗缺失，限制了策略的直接执行性。</p></html>"
    (paper/"problem2_4_report.html").write_text(html, encoding="utf-8")
    report2 = p2/"report"; report3 = p3/"report"
    report2.mkdir(exist_ok=True); report3.mkdir(exist_ok=True)
    (report2/"problem2_report.md").write_text("问题二详细正文见 paper/problem2_draft.md；可复查表格见 tables/。价格只在历史支持的加成率范围搜索，所有成本和需求模型采用时间顺序的 7 日回测。", encoding="utf-8")
    (report3/"problem3_report.md").write_text("问题三详细正文见 paper/problem3_draft.md；候选、策略、约束检查及敏感性结果均在 tables/。", encoding="utf-8")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\simhei.ttf")
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    daily_profit = plan.groupby("date", as_index=False).expected_profit.sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(daily_profit.date.astype(str), daily_profit.expected_profit, marker="o")
    ax.set(title="Problem 2 expected daily profit", xlabel="Date", ylabel="CNY")
    ax.tick_params(axis="x", rotation=30); fig.tight_layout()
    fig.savefig(p2/"figures/daily_expected_profit.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for category, x in plan.groupby("category"):
        ax.plot(x.date.astype(str), x.recommended_price, marker="o", label=category)
    ax.set(title="Problem 2 recommended category prices", xlabel="Date", ylabel="CNY/kg")
    ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(p2/"figures/category_price_paths.png", dpi=160); plt.close(fig)
    summary = item_plan.groupby("category").expected_profit.sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(summary.index, summary.values)
    ax.set(title="Problem 3 expected profit by category", xlabel="CNY")
    fig.tight_layout(); fig.savefig(p3/"figures/category_profit.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    product, sales, cost, loss = read_inputs(args.input_dir)
    normal, daily, cost_filled = build_daily(product, sales, cost, loss)
    models, demand_models = select_demand_models(daily)
    cost_fc, cost_models = cost_forecasts(daily)
    plan, baseline = optimize_categories(daily, models, cost_fc)
    cat_sens = category_sensitivity(plan, models, daily)
    candidates, options, candidate_week = item_demand_and_options(normal, cost_filled, loss, models)
    targets = plan[plan.date.eq(TARGET_DATE)].set_index("category").expected_demand_kg.to_dict()
    capacity = (options.sort_values("effective_inventory").groupby(["category", "item_id"], as_index=False)
                .tail(1).groupby("category", as_index=False).effective_inventory.sum()
                .rename(columns={"effective_inventory": "maximum_candidate_effective_inventory"}))
    capacity["problem2_target"] = capacity.category.map(targets)
    capacity["gap"] = capacity.maximum_candidate_effective_inventory - capacity.problem2_target
    debug_dir = args.out_dir/"analysis/problem3/tables"; debug_dir.mkdir(parents=True, exist_ok=True)
    capacity.to_csv(debug_dir/"候选供给能力诊断.csv", index=False, encoding="utf-8-sig")
    item_plan, result = solve_item_milp(options, targets)
    checks = [{"check": "selected_item_count_27_to_33", "value": len(item_plan), "passed": 27 <= len(item_plan) <= 33},
              {"check": "all_selected_replenishment_at_least_2_5kg", "value": float(item_plan.replenishment.min()),
               "passed": bool((item_plan.replenishment >= 2.5 - 1e-8).all())},
              {"check": "milp_success", "value": result.message, "passed": bool(result.success)}]
    for category, target in targets.items():
        actual = item_plan.loc[item_plan.category.eq(category), "effective_inventory"].sum()
        checks.append({"check": category + "_effective_inventory_ge_target", "value": actual-target,
                       "passed": bool(actual + 1e-6 >= target)})
    checks = pd.DataFrame(checks)
    if not checks.passed.all():
        raise AssertionError(checks.to_string(index=False))
    item_plan["expected_profit"] = item_plan["profit"]
    item_plan = item_plan.sort_values(["category", "expected_profit"], ascending=[True, False])
    sensitivity = item_sensitivity(options, targets, item_plan)
    write_markdown_reports(args.out_dir, plan, baseline, demand_models, cost_models, cat_sens,
                           candidates, item_plan, checks, sensitivity)
    (args.out_dir/"analysis/problem2/outputs.json").write_text(json.dumps({
        "category_profit": float(plan.expected_profit.sum()), "item_profit": float(item_plan.expected_profit.sum()),
        "candidate_items": int(len(candidates)), "selected_items": int(len(item_plan))
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"category_profit": float(plan.expected_profit.sum()),
                      "item_profit": float(item_plan.expected_profit.sum()),
                      "selected_items": int(len(item_plan))}, ensure_ascii=False))


if __name__ == "__main__":
    main()


# 2023 C 题问题二至四：可复现运行说明

## 输入

将四个原始附件放在同一个目录：`附件1.xlsx`、`附件2.xlsx`、`附件3.xlsx`、`附件4.xlsx`。程序只把 2023-06-30 及以前的记录用于拟合、回测和决策；问题二预测期为 2023-07-01 至 07-07，问题三决策日为 2023-07-01。

## 运行

```powershell
python -m pip install -r src/problem2_4/requirements.txt
python src/problem2_4/run_problem2_4.py --input-dir <附件目录> --out-dir .
```

程序读取正常销售记录建立需求口径。退货不进入需求建模；批发价缺失仅按单品使用此前观测值向前填补，并保留插补标记，避免使用未来信息。

## 结果与检查

程序重新生成 `analysis/problem2/`、`analysis/problem3/` 和 `paper/` 下的表格、图、草稿和 HTML。问题二的需求/成本模型只以时间顺序的滚动 7 日块验证；问题三自动断言选中 27--33 个商品、每个补货量不小于 2.5 kg、MILP 成功和每个品类有效库存达到目标。若任何断言失败，程序会中断而不会静默输出策略。

图表由 Matplotlib 输出为 PNG；完整浏览入口为 `paper/problem2_4_report.html`。所有价格搜索均限制在历史加成率支持范围内，模型表达的是运营预测关系，不作因果解释。


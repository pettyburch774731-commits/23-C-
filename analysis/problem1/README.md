# 问题一 EDA

在仓库根目录安装依赖后运行：

```powershell
python analysis/problem1/problem1_eda.py
```

脚本从 `问题1_建模数据` 读取单品日表和品类日表；若审计分支的
`analysis/problem1_audit_outputs/单品_活跃期统计.csv` 存在，会自动读入并
合并活跃期字段。输出表格、图和 `problem1_eda_report.md` 均写入本目录。

如果需要指定输入或输出目录，可使用 `--item-daily`、`--category-daily`、
`--active-stats` 与 `--analysis-dir` 参数。


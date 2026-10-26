# 实验结果

此目录保存可版本控制的性能实验摘要。建议按以下结构组织：

```text
results/
└── <topic>/
    ├── <YYYY-MM-DD>-<experiment>.md
    └── <machine-readable-result>.json
```

新实验从 [`docs/experiment-template.md`](../docs/experiment-template.md)复制模板。
benchmark 的 `--output-json` 可把统一 schema 的原始计时和环境信息写到本目录。
体积较大的 `.ncu-rep`、`.nsys-rep`、SQLite 和临时导出文件放到被忽略的
`artifacts/`，摘要中记录对应文件名即可。

提交摘要前确认它包含环境、Git 状态、工作负载、正确性、完整命令、关键计数器和
适用范围。没有这些信息的单个延迟或 TFLOP/s 数值不可复现。

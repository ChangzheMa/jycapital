# v2 四频 TCN 平台材料

来源运行：`v2-20261011-s42`。本地判定：**未达到本地改善条件**。导出用途：`user_requested_platform_test`；理由：用户明确要求为 v2 生成平台测试材料；本地未超过 v1，保留原判定，用于验证平台运行和官方评分。。本地结论与指标保存在 `local-evaluation.json`，不因导出而改变；这不是官方平台成绩。

将本目录所有文件放在平台同一目录，选择唯一的 `v2-submission.ipynb`。不要上传本地 `dai.py` 或行情。推理使用平台 DAI 和 `datasources` 中的四种频率映射，不联网、不训练、不读取收益标签。

`best.pt` 为冻结权重与模型配置；四个 `normalization-bar*.json` 和 `preprocessing-config.json` 为冻结预处理参数。`v2_model.py`、`v2_data.py`、`v2_preprocessing.py` 与正式训练时源码一致。`v2_predict.py` 是独立冻结的原始行情推理接口，验证结果以 `submission-validation.json` 为准。Notebook 末尾有带主程序保护的两日调试单元格；作为模块导入不执行行情查询，文件中不保存执行输出。

每只股票使用 1／5／15／30 分钟的 60／24／16／16 个常规交易时间槽，72 个 z/c 值、72 个缺失掩码以及时间编码，四分支因果 TCN 输出一个因子。缺参股票按截至该时点的可用历史更新统计量。推理开始前只为缺参股票查询训练期之后的历史，分日处理完整截面以限制内存。

推理依赖见 `requirements.txt`，支持 CUDA 时使用 GPU，否则使用 CPU；实际平台依赖和性能须在平台验证。训练附加依赖见 `requirements-training.txt`，复现需先准备授权原始 Parquet 数据。

```bash
python v2_reproduce.py --workspace reproduced-v2 --data-root /path/to/authorized/data
```

该命令要求一个尚不存在的工作目录，按原仓库结构还原所附源码，重新拟合 2020—2023 年统计量、生成验证／测试输入并进行一次正式训练。产生大量本地中间数据，需要充足磁盘与训练资源。跨硬件或依赖版本不保证逐位相同。

`model-config.json`、`protocol.json`、`selection.json` 记录 seed、超参数、日期和权重选择依据。`submission-manifest.json` 包含全部文件哈希；`submission-validation.json` 记录实际完成的本地验证范围。比赛平台由用户提交，平台运行和官方评分另行确认。

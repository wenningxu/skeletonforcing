# Joint-Time Valley Motion Forcing

研究代码、配置、测试与实验说明。约束及实验授权范围见 [AGENTS.md](AGENTS.md)。

当前 OF009 准备方案采用标准 Wan、266D、无文本、无 motion VAE，默认从第 0 帧根关节向时间和骨架传播。每个关节帧采用平移后的同一条 32 步降噪曲线；64 帧共 53 次模型调用，完成的状态冻结，无真实根姿态输入。当前只完成 CPU 检查，尚未执行 OF009 GPU 实验，不能据此宣称生成质量、流式延迟或调度优势。

- [当前方案：等步数 Valley](reports/run_report_v11_equal_steps_valley_zh.md)
- [Wan 组件标准化](reports/wan_standardization_v1_zh.md)
- [OF008 实验结果](reports/OF008_results_zh.md)
- [跨电脑同步与环境配置](docs/GITHUB_SYNC_zh.md)

OF001–OF008 的历史报告保存在 `reports/`。已完成的授权不可重复用于旧实验。GPU 操作必须遵守 AGENTS.md 中最新授权范围；克隆仓库不代表可以启动付费资源。

CPU 测试：

```powershell
python -m pytest tests -q
```

数据、模型权重、实验输出、准备状态库、虚拟环境及凭据不进入 GitHub。完整复现实验还需要报告列出的外部资产；网络卷中的检查点不会随 Git 同步。第三方源码保存在 `third_party/`，保留其许可证和版本记录。

# 跨电脑接续：2026-09-30

源项目：D:\motionskeletonforcing。目标电脑：desktop-4mm681h（Windows）。
本文件是研究状态交接，不是 GPU 实验启动指令，也不是完整聊天记录。

## 新电脑上的第一条消息

> 接续这项动作生成研究。请先阅读 docs/MIGRATION_HANDOFF_2026-09-30.md、AGENTS.md、reports/run_report_v11_equal_steps_valley_zh.md，以及 configs/root_first_overfit_v1.json。先核验迁移文件、重建本地 CPU 环境、检查 Runpod 连接和已有云盘，只做接续检查，不启动 GPU。当前是 OF009 等步数山谷时间场：从第 0 帧 root 出发，沿时间和骨骼链传播；每个关节帧恰好 32 次更新，错峰开始、错峰完成。确认状态后向我汇报。

## 当前研究状态（优先于旧 README）

- OF001—OF008 已结束，结果在 reports 与 outputs。AGENTS.md 中历史 ACTIVE 段被后续 FINAL 段覆盖，不能重新启动这些实验。
- 最近一次研究记录确认任务 Pod 已删除；本次迁移没有创建、重启或使用 GPU。新电脑连接 Runpod 后做一次只读清单确认当前状态。
- OF009 仅完成代码、CPU 验证和可视化，未启动 GPU。最新报告是 reports/run_report_v11_equal_steps_valley_zh.md，v10 两版均已过时。
- 原根目录 README 的“尚未做真实数据过拟合”及 v2 链接是历史内容；不要据此覆盖后续实验状态。
- 当前不做控制、不用文本、不用 motion VAE。采用标准 Wan joint-time backbone，78,053,389 参数，width1024/depth8/heads8/FFN2048，逐 token、逐层 sigma AdaLN。
- 保留 266D 冗余表示：root4 + world XYZ66 + rotation6D126 + velocity66 + contact4；不能简化成 22 XYZ。
- 22 个独立关节，不是 8 个骨骼块。图中有 8 个深度层。
- 调度入口 (frame0,root0) 仅用于计算距离，不提供 GT。known mask 全 false，所有通道从 Gaussian 开始。已完成节点冻结的是模型生成状态。
- 距离 D=f/20+root_graph_distance(j)/2；onset=ceil(32*0.75*D/(1+D)-1e-6)。sigma(k)=1-clamp(k-onset,0,32)/32。
- 每个关节帧所有所属通道都有同样的 32 次有效更新。64 帧对应 53 次全局模型调用、54 个状态。frame0/root 完成于32，frame0/wrist于51，最后帧/wrist于53。不能恢复成所有节点同时归零。
- 训练取采样轨迹中的完整时间场；先400步 teacher，随后50%实际模型 rollout，前缀 detach、最后4步带梯度；速度一致性要屏蔽未开始的相邻位置及未就绪 heading。
- 仍然是整段64帧全注意力，不宣称因果流式生成。CPU 正确性不证明动作质量或调度优势。

## 当前代码和证据

- motion_valley/root_first.py：调度、通道归属和更新。
- motion_valley/root_first_experiment.py：训练/评估入口。
- configs/root_first_overfit_v1.json：schedule_version=shifted_equal_steps_v2，sampling_steps=53、local_steps=32。
- reports/of009_source_audit.json：42个源文件的 SHA256；报告、配置、审批文件还有绑定哈希。迁移不应改源文件换行。
- reports/local_checks_root_first.json：此前56个 CPU tests 通过，小 Wan batch3×64 BF16 52步前缀/4步梯度尾通过；不是在新电脑复测的结果。
- outputs/OF009-equal-steps/default_valley_denoising.gif：最新动图，54状态；旁边有 time_field.png、manifest 和实际 sigma 数组。OF009-preflight 的旧动图是历史版本。
- prepared/OF003/states.npz：本地小样本数据；不要误认成 HumanML3D 全量数据。
- approvals/of009-default-valley.json 目前 approved=false，准备阶段记录，不应为迁移而修改它。

## 研究结论边界与授权

OF006 的传统 inpainting 改善重建，但未证明可靠控制；OF007 增加密度/数据/布局的有限测试仍未达到控制门槛；OF008 的 Kimodo266 适配版出现严重时间塌缩，不能推出“架构无关、必然只是数据量问题”。详见各 results_zh 报告。后续已回到无控制时间场顺序生成方案。

总预算 USD500 不是任意启动实验的授权。用户对既有标准 Wan 的有界过拟合/少样本测试给予免逐次询问授权，但仍需新报告、资源上限和运行记录；全量训练、不同模型概念/骨干、noisy anchors 等不在该范围。本次迁移不触发任何运行。不要复用已消费的 OF001—OF008 审批。

## Runpod 资源与配置

- 保留现有账户中的私有网络卷 k1j34iuc97，100GB，EU-RO-1。HumanML3D、官方 UMT5/evaluator 资产、各次完整 checkpoint 在云卷上，迁移电脑无需复制或重建此卷。BABEL 未准备好。
- checkpoint 一般仅在云卷 /workspace/outputs/OFxxx/...，本地主要是证据。具体路径和哈希见 reports/resources_of*.json。
- reports/pod_connection.json 及旧 SSH helper 中的 Pod/IP 是失效历史信息，不应直接连接或重启。
- 源电脑 MCP endpoint 为 https://mcp.getrunpod.io/，注册名 runpod；本机未发现 ~/.runpod/config.toml、config.yaml 或进程 RUNPOD_API_KEY。OAuth 登录在新电脑重新完成，不复制整个 Codex auth/config/数据库。
- Runpod 插件来源：https://github.com/runpod/runpod-plugins-official.git；源电脑插件 runpod@runpod 1.2.0。新电脑安装对应插件即可恢复技能；不要照搬源电脑插件缓存绝对路径。
- SSH 私钥与项目包分开迁移（如用户选择保留），按另附说明导入。不要把密钥粘贴进聊天，也不要把它上传到代码仓库或普通证据包。

在新电脑的 Codex 终端中执行（先确认没有同名冲突配置）：

```powershell
codex mcp add runpod --url https://mcp.getrunpod.io/
codex mcp login runpod
codex plugin marketplace add https://github.com/runpod/runpod-plugins-official.git
```

浏览器登录原 Runpod 账户，然后在 Codex 插件界面安装/启用 Runpod、开启新任务加载工具。若插件已自动提供并成功连接同一 MCP，则无需重复添加。CLI 不在 PATH 时可以让新电脑 Codex 按本机安装路径执行。这些操作不创建云资源。

## 接收与恢复

本次用 Tailscale Taildrop 传送一次性快照，不是持续双向同步。先在目标电脑的 Tailscale 接收目录寻找 motionskeletonforcing-20260930.zip、SHA256 文件和此说明；若还在收件箱，可运行：

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\Downloads\motion-transfer"
tailscale file get --conflict=rename "$env:USERPROFILE\Downloads\motion-transfer"
```

必要时管理员 PowerShell 执行 Tailscale 命令。对照 SHA256 文件运行 Get-FileHash -Algorithm SHA256 核验 ZIP。解压到空目录；例如解压到 D:\ 后得到 D:\motionskeletonforcing。不要覆盖另一份已有研究副本。ZIP 内 MIGRATION_MANIFEST.json 提供逐文件 SHA256、排除项及 Python 包版本。

未复制 .venv、缓存、.secrets、.partial 下载及 Codex 会话数据库。普通项目 ZIP 有代码、第三方源码、准备数据、报告、配置、已完成实验的本地证据及动图。

新电脑安装 Python 3.12，进入解压后的项目根目录，重建环境：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt matplotlib pillow
.venv\Scripts\python.exe -m pytest tests -q
```

requirements 使用版本范围；精确源环境版本记录在 MIGRATION_MANIFEST.json，若遇到兼容差异以此比较。CPU 检查不租用 GPU；不要直接运行历史 run_of*.sh。目标路径若不是 D:\motionskeletonforcing，应检查旧 helper 里的硬编码本地路径，按需修改并重做运行前哈希绑定。

在新电脑 Codex 中把解压目录添加为项目，再发送顶部的接续提示。文件包保存工作状态与证据，不承诺将原聊天逐条导入侧栏。原电脑保留原对话和项目作为备份；后续只在一台电脑上修改项目，避免分叉。

参考：
- https://learn.chatgpt.com/docs/extend/mcp?surface=cli
- https://tailscale.com/docs/features/taildrop

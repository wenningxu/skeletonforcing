# GitHub 跨电脑同步

仓库：https://github.com/wenningxu/skeletonforcing.git

## 新电脑

安装 Git 与合适的 Python/PyTorch 环境，并登录具有仓库写入权限的 GitHub 账号。首次下载：

```powershell
git clone https://github.com/wenningxu/skeletonforcing.git
cd skeletonforcing
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest tests -q
```

PyTorch 的 GPU 构建应与机器驱动匹配。运行实验前阅读 AGENTS.md、当前报告和配置；旧连接记录不可复用。数据、UMT5/评估器资产、准备状态库及检查点需要独立获取，Git 克隆不会包含它们。凭据在每台电脑单独配置，不能提交到仓库。

## 日常同步

开始工作前，在工作区干净时拉取：

```powershell
git pull --ff-only
```

完成一个重大且完整的修改后，检查差异、执行必要验证，再提交并推送：

```powershell
git status
git diff
git add <本次修改的文件>
git diff --cached
git commit -m "Describe the completed change"
git push origin main
```

不用每个操作都提交。以完成的功能、协议修改、实验结果报告或重要文档更新为提交单位。换电脑前推送，另一台电脑开工前拉取。若有未提交修改或分支分叉，先保存并处理冲突；不要强推或直接覆盖本地文件。

这是手动 Git 同步约定，不是后台定时同步。后续本聊天中的重大修改按 AGENTS.md 约定提交推送。实验授权和付费资源审批不因同步改变。

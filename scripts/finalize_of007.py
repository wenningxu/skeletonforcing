"""Write the local OF007 result report from audited evidence; no remote actions."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
analysis = json.loads((ROOT / 'outputs/OF007-analysis/analysis.json').read_text())
models = analysis['models']
resource_path = ROOT / 'reports/resources_of007.json'
resource = json.loads(resource_path.read_text())
assert resource['evidence_files_verified'] == 148
assert analysis['audited_common_protocol_samples'] == 640
assert analysis['audited_routine_samples'] == 100
runlog = (ROOT / 'outputs/OF007-run.log').read_text()
assert '42 passed in 8.88s' in runlog
all_rows = []
for model in models:
    all_rows += [json.loads(line) for line in (ROOT / f'outputs/OF007/{model}_eval/evaluation.jsonl').read_text().splitlines()]
    if model != 'baseline':
        all_rows += [json.loads(line) for line in (ROOT / f'outputs/OF007/{model}/evaluation.jsonl').read_text().splitlines()]
assert len(all_rows) == 1040 and all(r['all_steps_known_exact'] for r in all_rows)
max_anchor = max(r['anchor_max_m'] for r in all_rows)
confirmed = '2026-09-26T06:09:52Z'
elapsed = (datetime.fromisoformat(confirmed) - datetime.fromisoformat(resource['created_at'])).total_seconds()
resource.update(status='DELETED', live_stage='COMPLETED; local evidence and sample audit passed; no active or queued jobs',
    deletion_http_status=204, zero_pods_confirmed_at=confirmed, gpu_cost_upper_estimate_usd=elapsed / 3600 * .74,
    billing_read_at='2026-09-26T06:09:12Z', billing_posted_gpu_usd=.12148970365524292,
    billing_posted_disk_usd=.00069444440305233, billing_posted_total_usd=.12218414805829525,
    billing_final=False, billing_note='Partial posted billing; final total pending; network-volume storage separate',
    pod_environment_tests_passed=42, audited_samples=740, evaluated_cases=1040,
    primary_response_pass_counts={m:r['primary']['response_pass_count'] for m,r in models.items()},
    max_physical_anchor_error_m=max_anchor, network_volume_preserved=True,
    result_report='reports/OF007_results_zh.md')
resource_path.write_text(json.dumps(resource, indent=2) + '\n')
connection_path = ROOT / 'reports/pod_connection.json'
connection = json.loads(connection_path.read_text())
assert connection['pod_id'] == resource['pod_id']
connection.update(status='DELETED', stale=True, zero_pods_confirmed_at=confirmed)
connection_path.write_text(json.dumps(connection, indent=2) + '\n')

labels = {'baseline':'冻结 OF006 基线', 'density':'控制点 3→15', 'data':'动作 1→8 条', 'coverage':'训练布局 1→3'}
def metric(model, key, scale=1):
    return models[model]['primary'][key]['mean'] * scale

lines = ['# OF007 结果：分别增加控制点、动作数量与布局覆盖', '',
'2026-09-26。按用户“好的，构建实验，通过”执行；三个新模型各训练一次、各 4,000 步，冻结 OF006 基线重新评价。全部执行完成，GPU 已删除，原网络卷与 checkpoint 保留。', '',
'## 结论', '',
'三项干预均未达到预定有效控制门槛。增加控制点和动作数量出现了小幅、可测的响应；增加布局主要改善重建。本轮不能支持“数据少就必然无法控制”，也不能支持“扩大数据无效”。它表明：在当前训练目标、归一化和固定 4,000 步预算下，单独做这三种扩容都不足以解决控制响应不足。', '',
'## 实际方案与公平比较', '',
'所有模型使用同一 78,053,389 参数无文本 Wan 骨干（1024 宽、8 层、8 头、FFN 2048），全关节时间注意力、Q/K RMSNorm、三轴 RoPE、逐 token 逐层时间 shift/scale/gate，motion+mask 显式线性拼接。266D 表示保留 root、世界 XYZ、6D 旋转、局部速度和接触；无运动 VAE。', '',
'沿用传统 inpainting 的统一未知噪声场 σ=1−k/32，锚点 XYZ 全过程 σ=0，每次输入及更新回填。训练使用该推理轨迹的完整时间场，前 400 步 teacher，之后 50% 模型 rollout；rollout 前缀截断，最后 4 步保留梯度。目标为未知通道 x0 MSE 加 0.01 冗余一致性损失。本轮不是谷地时间场优势试验，也没有噪声锚点、额外响应损失或新控制分支。', '',
'| 模型 | 动作 / 训练状态 | 每例控制点 | 训练布局 | 训练步数 |',
'|---|---:|---:|---|---:|',
'| baseline | 1 / 3 | 3 | A | 冻结历史 4,000 步 |',
'| density | 1 / 3 | 15 | A15 | 4,000 |',
'| data | 8 / 24 | 3 | A | 4,000 |',
'| coverage | 1 / 3 | 3 | A/B/C，每批选一个 | 4,000 |', '',
'主比较全部在同一动作 011936 上进行。density 用 A15/D15 对比 baseline 的 A3/D3；data、coverage 使用 A3/D3。共同评分区域是帧 36–60、关节 17/19/21，并统一去除 A3/A15/D3/D15 全部锚点的并集，剩 48 个关节帧位置，避免密集固定点直接获得响应分数。D/D15 没有参与训练。', '',
'每模型四种推理布局 × 五种状态角度 × 八个噪声，160 例，共 640 例。训练角度 −12/0/+12°，未训练角度 ±6°；改变控制时与 0°基准使用相同初始噪声，跨模型也复用噪声。各模型预定主比较为 80 例，其中 64 个非零角度响应；另外两个推理布局只作次要诊断。', '',
'## 共同区域主结果', '',
'令 a 为改变控制前后生成结果的自由区域差，b 为对应 GT 差。gain=<a,b>/<b,b>，relative error=||a−b||/||b||。正确响应为 gain=1、error=0；无响应为 gain=0、error=1。通过门槛：gain∈[0.5,1.5] 且 error<0.5；每布局/角度组至少 6/8 噪声通过。', '',
'| 模型 | gain ↑（理想 1） | 相对误差 ↓（理想 0） | 实际变化 RMS mm | 响应通过 | 分组通过 |',
'|---|---:|---:|---:|---:|---:|']
for m, r in models.items():
    lines.append(f"| {labels[m]} | {metric(m,'common_response_gain'):.6f} | {metric(m,'common_response_relative_error'):.6f} | {metric(m,'common_response_actual_rms_m',1000):.3f} | {r['primary']['response_pass_count']}/64 | {r['primary_groups_passed']}/8 |")
lines += ['', '四组对应 GT 变化 RMS 都为 22.414 mm。数据组虽然有约 1.9% 的沿 GT 方向增益，但实际变化仍不到 1 mm，远低于目标，不能称为控制成功。所有主比较分组都失败，包括已训练布局与已训练角度的组合，所以问题不只出现在未见控制条件的泛化上。', '',
'| 模型 | 共同 48 点 MPJPE mm | 自身自由区域 MPJPE mm | FK/XYZ 不一致 mm | 重建门槛通过 |',
'|---|---:|---:|---:|---:|']
for m, r in models.items():
    lines.append(f"| {labels[m]} | {metric(m,'common_region_mpjpe_m',1000):.3f} | {metric(m,'free_xyz_mpjpe_m',1000):.3f} | {metric(m,'fk_position_disagreement_m',1000):.3f} | {r['reconstruction_pass_count']}/80 |")
lines += ['', '自身自由区域随控制点密度改变，其 MPJPE 不能替代共同区域比较。coverage 的自由 MPJPE 从 9.781 mm 降至 5.710 mm，但响应 gain 仍只有 0.000446；全局重建通过显然不代表学会编辑。data 在原动作上重建明显恶化，尚未达到多动作充分拟合。', '',
'![配对控制响应](../outputs/OF007-analysis/paired_response.png)', '',
'图中虚线为目标变化；横轴角度仅用于构造/评价监督状态，不作为模型输入。误差棒为八个采样噪声结果的标准差。', '',
'## 小幅变化的不确定性', '',
'按噪声 seed 整块重采样 4,000 次，将同一 seed 的角度和布局一起保留，比较各模型相对冻结基线的平均 gain 差。下列区间为三个比较经 Bonferroni 调整的 98.333% bootstrap 区间。', '',
'| 干预 | gain 差 | 区间 |', '|---|---:|---|']
for m in ['density','data','coverage']:
    d = models[m]['gain_delta_vs_baseline']; lo, hi = d['bootstrap_family_ci_98_333']
    lines.append(f"| {labels[m]} | {d['mean']:.6f} | [{lo:.6f}, {hi:.6f}] |")
lines += ['', 'density/data 的区间高于零，但实际增益极小；coverage 包含零。这些区间只描述固定 checkpoint 的采样噪声不确定性。每组只有一个训练种子，八个采样种子不能替代八次训练，不能声称已证明跨训练种子的显著提升。', '',
'## 次要诊断与完整指标', '',
'四种布局均使用相同 48 点评分；以下为平均 gain，不能用次要组替换预定主结果。', '',
'| 模型 | A3 | A15 | D3 | D15 |', '|---|---:|---:|---:|---:|']
for m, r in models.items():
    lines.append('| '+labels[m]+' | '+' | '.join(f"{r['by_layout'][l]['common_response_gain']['mean']:.6f}" for l in ['A3','A15','D3','D15'])+' |')
lines += ['', '密集训练模型在稀疏推理时响应又接近零；数据组在额外密集条件下略有增益，但仍远低于门槛。这里只说明当前 checkpoint 对更多控制信息产生了微弱变化，不能外推任意关节编辑能力。', '',
'训练入口原有常规评价另有 400 例：density 40、data 320、coverage 40；其中部分条件与共同协议重复，不是额外 400 个独立统计样本。其自由区域仅排除各自锚点，以下结果是次要指标。', '',
'| 模型 | 重建通过 | 响应通过 | 自由 MPJPE mm | 旋转归一化 MSE | 速度归一化 MSE | 骨长 MAE mm | 速度误差/静态基线 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for m in ['density','data','coverage']:
    r=models[m]['routine']; s=r['statistics']; groups=[v for v in r['summary'].values() if isinstance(v,dict)]
    lines.append(f"| {labels[m]} | {sum(g['reconstruction_pass_count'] for g in groups)}/{s['count']} | {sum(g['response_pass_count'] for g in groups)}/{sum(g['response_count'] for g in groups)} | {s['free_xyz_mpjpe_m']['mean']*1000:.3f} | {s['rot6d_normalized_mse']['mean']:.6f} | {s['velocity_normalized_mse']['mean']:.6f} | {s['bone_length_mae_m']['mean']*1000:.3f} | {s['velocity_error_vs_static_ratio']['mean']:.4f} |")
lines += ['', 'data 八条训练动作全部未通过重建与响应门槛。其常规响应 gain 均值 0.238、相对误差均值 1.395，包含异常大变化：最大 gain 25.585、最大相对误差 56.631；不能将这种不稳定响应称为更强控制。逐动作汇总：', '',
'| bank 动作索引 | 自由 MPJPE mm | gain | 相对误差 |', '|---|---:|---:|---:|']
for clip,s in models['data']['routine']['by_clip'].items():
    lines.append(f"| {clip} | {s['free_xyz_mpjpe_m']['mean']*1000:.3f} | {s['response_gain']['mean']:.6f} | {s['response_relative_error']['mean']:.6f} |")
lines += ['', '最后 1,000 步中已记录训练点的重建损失均值如下，teacher 与 rollout 所处时间步不同，不能将二者差直接解释为唯一失败原因。', '',
'| 模型 | teacher MSE / 记录数 | rollout MSE / 记录数 | 训练秒数 | 峰值 allocated GB |',
'|---|---:|---:|---:|---:|']
for m in ['density','data','coverage']:
    r=models[m]; t=r['last_1000_logged_training_steps']; a=t['teacher']; b=t['rollout']
    lines.append(f"| {labels[m]} | {a['mean']:.6f} / {a['logged_count']} | {b['mean']:.6f} / {b['logged_count']} | {r['train_seconds']:.2f} | {r['peak_allocated_gb']:.3f} |")
lines += ['', '## 能排除什么，尚不能归因什么', '',
'1. “锚点回填得准”不能解释本轮分数：所有固定点都排除出共同区域。训练条件向量经 CPU 核对彼此不同，最小欧氏间距为 density 128.50 mm、data 78.69 mm，coverage 的 A/B/C 为 78.69/51.86/34.36 mm。不存在这批训练样本的控制向量完全相同、却要求不同监督的直接冲突；这不等于模型已经学会利用条件。',
'2. 单纯增加布局不能解决当前响应不足；增加密度和数据量有小幅变化，但不足以使本轮模型可控。不能由有限干预推断任意规模数据、任意训练时长的结论。',
'3. data 采用等步数比较：每组都呈现 12,000 个训练样本，八动作使每个状态平均呈现次数从 4,000 降至 500。此外为保持尺度不变，固定了单动作归一化，新动作归一化 XYZ 最大绝对值达 24.20，而原动作为 2.76。因此其恶化同时受到学习难度、每状态训练量及尺度适配的影响；尚不能把失败唯一归因为数据量。',
'4. density 新增点来自同一个一维手臂扰动，增加观测密度但没有增加独立控制自由度。主试验只有一个参考动作；data 的其它七条也都是训练动作，没有未见动作泛化证据。',
'5. 当前证据更值得优先检查“局部条件变化在整体重建目标中的学习压力及优化是否足够”。这是下一步工作假设，尚无响应损失、局部加权或梯度路径的干预证据。本轮不自动追加训练。', '',
'## 验证与协议边界', '',
f'本地与 Pod 环境各 42 项测试通过。完整 1,040 次生成均记录逐步已知坐标完全一致；反归一化世界坐标的最大锚点误差为 {max_anchor:.10g} m，属于数值精度量级。硬约束满足与 FK/速度一致性、生成质量、控制响应分别报告。', '',
'640 份共同协议样本全部保存；常规协议保存首个噪声的 100 份样本。共 740 份保存样本用 NumPy 独立重算自由 MPJPE、锚点误差与响应指标，并检查共同协议的 mask、GT、shape 和有限性。逐步性质依据 GPU 采样回调日志与测试，未保存每个中间状态供逐步回放。148 个证据文件 SHA256/长度、外层压缩包 SHA256、配置与冻结源码均核验通过。', '',
'本轮是少样本机制诊断，未运行 FloodDiffusion 官方完整 FID、文本匹配、BABEL、流式延迟或全数据协议。沿用现有 HumanML3D 衍生 266D 数据和一致性诊断，不能据此声称领先该论文。本轮样本是运动学一致的局部旋转构造，未验证动力学、碰撞、语义；未新增 XYZ→官方特征转换评估或转换误差底线测量。', '',
'审批报告和 GPU 运行源码在开跑前冻结；本地结果分析/报告生成脚本于结果下载前后补充，未修改训练、评估结果和预定主判据。完整配置、每个条件结果和检查点哈希保存在相邻记录中。', '',
'## 资源与可复现文件', '',
f"Pod ubz5b4x1f4ya91，RTX4090，$0.74/小时，创建于 {resource['created_at']}。删除返回 HTTP 204，零 Pod 已确认并于 {confirmed} 记录；按该较晚时间保守计的 GPU 费用上界约 **${resource['gpu_cost_upper_estimate_usd']:.4f}**，低于本轮 $2 上限。已返回账单仅 $0.122184（GPU $0.121490、盘 $0.000694），尚不完整，不能当最终费用。网络卷存储另计。", '',
'网络卷 k1j34iuc97 保留；三个完整 checkpoint 各约 936.8 MB，仅在卷上，未下载本地。旧 SSH 记录已标记失效，没有活跃或排队任务。', '',
'| 模型 | 卷内 checkpoint | SHA256 |', '|---|---|---|']
for m,c in resource['checkpoints'].items():
    lines.append(f"| {m} | `{c['path']}` | `{c['sha256']}` |")
lines += ['', f"结果包 SHA256：`{resource['evidence_sha256']}`。", '',
f"GPU 源码包 SHA256：`{resource['source_archive_sha256']}`。", '',
'- [运行前方案](approval_report_v8_zh.md)',
'- [资源、费用与检查点记录](resources_of007.json)',
'- [机器可读完整汇总](../outputs/OF007-analysis/analysis.json)',
'- [实验计划](../configs/control_factors_v1.json)',
'- [评价实现](../motion_valley/control_factors.py)',
'- [本地审计与统计实现](../scripts/analyze_of007.py)', '']
(ROOT/'reports/OF007_results_zh.md').write_text('\n'.join(lines),encoding='utf-8')
agents=ROOT/'AGENTS.md'
marker='OF007 FINAL (2026-09-26):'
if marker not in agents.read_text(encoding='utf-8'):
    with agents.open('a',encoding='utf-8') as f:
        f.write(f'\n{marker} supersedes ACTIVE paragraph. All3 once-only arms completed4000steps; frozenOF006 reevaluated.640common+400routine generations,740saved samples locally audited,148filehashes verified,42local/Pod tests pass. Primary common48freepoints baseline/density/data/coverage gain=-0.00003180/0.00806788/0.01930831/0.00044563, error=1.000176/0.992142/0.981594/0.999708; all0/64response,0/8groups. Primary freeMPJPE9.781/8.338/130.857/5.710mm; reconstruction80/80,80/80,0/80,80/80. Density/data show small response increases for these fixed checkpoints only; no effective control. One training seed, equalcompute not equalepochs; fixedsingleclipnormalizer causes additionaldataXYZscale up to24.2, so do not infer moredata useless. No noisyanchor/fulltraining/scheduleadvantage claim. Pod ubz5b4x1f4ya91 DELETED,zero pods confirmed by{confirmed}; GPUupperestimateUSD{resource["gpu_cost_upper_estimate_usd"]:.6f}, partialpostedpodUSD0.122184, finalpending/storageseparate. Preservevolume k1j34iuc97; checkpoints /workspace/outputs/OF007/{{density,data,coverage}}/last.pt onlyonvolume withhashes inreports/resources_of007.json. Evidence outputs/OF007-evidence.tar.gz SHA256{resource["evidence_sha256"]}. See reports/OF007_results_zh.md and outputs/OF007-analysis. Connectionstale; noactive/queuedjobs, do not rerun OF007. Standing boundedoverfit authorization persists; further experiments need fresh concrete scoped reports.\n')
print(json.dumps({'report':'reports/OF007_results_zh.md','status':resource['status'], 'cost_upper':resource['gpu_cost_upper_estimate_usd'],'audited_samples':740,'max_anchor_error_m':max_anchor},indent=2))

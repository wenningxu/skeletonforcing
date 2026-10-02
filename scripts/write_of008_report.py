"""Write the completed-run report from audited local artifacts."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
load=lambda p:json.loads((root/p).read_text())
a=load('outputs/OF008-analysis/analysis.json');r=load('reports/resources_of008.json')
probe=load('outputs/OF008-analysis/cpu_condition_probe.json')
assert r['status']=='DELETED'
b=a['models']['baseline'];k=a['models']['kimodo266'];d=a['gain_difference']
mean=lambda x,key:x['primary'][key]['mean']
rows=[]
for label,v in [('Wan inpainting 基线',b),('Kimodo266 适配',k)]:
    rows.append(f"| {label} | {mean(v,'common_response_gain'):.8f} | {mean(v,'common_response_relative_error'):.6f} | {1000*mean(v,'common_response_actual_rms_m'):.3f} | {v['primary']['response_pass_count']}/64 | {v['groups_passed']}/8 |")
quality=[]
for label,v in [('Wan inpainting 基线',b),('Kimodo266 适配',k)]:
    quality.append(f"| {label} | {1000*mean(v,'free_xyz_mpjpe_m'):.3f} | {1000*mean(v,'common_region_mpjpe_m'):.3f} | {1000*mean(v,'fk_position_disagreement_m'):.3f} | {v['reconstruction_pass_count']}/80 |")
groups=[]
for name,s in k['groups'].items():
    groups.append(f"| {name} | {s['common_response_gain']['mean']:.8f} | {s['common_response_relative_error']['mean']:.6f} | {s['response_pass_count']}/8 |")
secondary=[]
for name in ['A15','D15']:
    s=k['by_layout'][name]
    secondary.append(f"| {name} | {s['common_response_gain']['mean']:.8f} | {s['common_response_relative_error']['mean']:.6f} | {s['response_pass_count']}/{s['response_cases']} |")
phys=[]
for key in ['feature_mse','rot6d_normalized_mse','velocity_normalized_mse','velocity_error_vs_static_ratio','bone_length_mae_m']:
    phys.append(f"| {key} | {a['routine'][key]['mean']:.8g} |")
temporal=[]
for name,v in a['posthoc_temporal_variation'].items():
    temporal.append(f"| {name} | {1000*v['prediction_temporal_rms_m']:.6f} | {1000*v['gt_temporal_rms_m']:.3f} | {v['temporal_rms_ratio']:.8f} |")
text=f'''# OF008 结果：Kimodo 控制注入与两阶段骨干的 266D 适配

2026-09-26。按用户“通过”批准的 v9 报告执行一次：4,000 步训练、80 例常规评估与 160 例共同协议评估全部完成。GPU 已删除，网络卷与完整 checkpoint 保留。

## 主要结果

Kimodo266 主比较有效响应通过 **{k['primary']['response_pass_count']}/64**、布局/角度分组通过 **{k['groups_passed']}/8**、重建通过 **{k['reconstruction_pass_count']}/80**。本轮没有建立控制能力改善。对结果的解释应同时考虑重建是否成功，不能仅由两个模型失败推出“架构已排除”或“只需增加训练量”。

本次输出出现明显的时间变化塌缩：排除控制点后，输出时间变化RMS仅 {1000*a['posthoc_temporal_variation']['kimodo266']['prediction_temporal_rms_m']:.3f} mm，GT为 {1000*a['posthoc_temporal_variation']['kimodo266']['gt_temporal_rms_m']:.3f} mm，即约 {100*a['posthoc_temporal_variation']['kimodo266']['temporal_rms_ratio']:.4f}%。示例手腕轨迹几乎为常值，仅锚点帧由硬回填产生跳变。这次更换骨干没有获得有效的基本动作拟合，因此不能作为排除架构因素的证据。

## 实际实施

直接加载 [Kimodo 官方仓库](https://github.com/nv-tlabs/kimodo) 固定提交 `58e781898b3d7e328a676a75d3e338c45dce3ad9` 的 TransformerEncoderBlock 和 TwostageDenoiser，未改写官方组件。采用官方两阶段帧级 Transformer：每阶段 16 层、1024 宽、8 头、FFN 2048、post-norm、GELU、零 dropout、正弦位置编码与时间前缀。50 个零内容前缀保留，但无文本编码器或 caption、无 GT heading。

沿用 root4 + world XYZ66 + rot6d126 + velocity66 + contact4 的 266D；无动作 VAE。已知 XYZ 回填当前状态，再拼接完整逐通道 mask，送入两个阶段。root 阶段输出 root4 与 pelvis XYZ3；body 阶段接收预测 local root4、带噪身体259D与mask，输出身体259D。保留官方训练时 root→body detach。root/旋转/速度未知值不作为额外 GT 条件输入。

这是现有 266D 与 x0 flow 协议上的适配，**不是完整原生 Kimodo 复现**。没有使用原生平滑 global-root 表示、预训练权重、分组 Smooth L1/FK 损失、课程、EMA、CFG或后处理。与 Wan 相比，token 粒度、两阶段划分、容量、时间注入及优化适配同时变化；不能视作单一控制注入因素的消融。参数量 282,752,266 对 78,053,389；相同步数/样本次数不等于相同 FLOPs。

本轮为传统 inpainting 对照，未知通道 sigma=1−k/32，已知 XYZ 全程 sigma=0；时间前缀索引 round(1000*sigma)。不验证关节谷地时间场的优势。训练使用同一推理格点，前400步 teacher，之后50%模型 rollout、前缀截断、最后4步保留梯度；未知通道归一化x0 MSE加0.01 root/速度一致性。AdamW lr=2e-4、betas=(0.9,0.99)、100步预热、梯度裁剪1、batch3、BF16 autocast。

训练数据为动作011936、64帧、−12/0/+12°三个状态，布局A；±6°和布局D未训练。从随机初始化训练，未冻结新模型。归一化与OF006基线逐元素相等。该几何变体库没有证明语义、碰撞或人体可行性，也不代替真实多动作数据。

## 固定噪声控制响应

主协议 A3/D3 × 五角度 × 八噪声，共80例，其中64个非零角度响应。改变控制时复用同一噪声，模型间复用同seed。共同区域为帧36–60、关节17/19/21，去除四种布局锚点并集后剩48个自由位置。基线取自已保存并审计的 OF007/baseline_eval，即历史 OF006 checkpoint。

gain=⟨生成变化,GT变化⟩/‖GT变化‖²；error=‖生成变化−GT变化‖/‖GT变化‖。理想值分别为1和0，无响应为0和1。门槛固定为 gain∈[0.5,1.5] 且 error<0.5；每布局/角度组至少6/8噪声通过。

| 模型 | gain | relative error | 实际变化RMS mm | 响应通过 | 分组通过 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

对应GT变化RMS为 {1000*mean(k,'common_response_target_rms_m'):.3f} mm。

| 模型 | 自由MPJPE mm | 共同区域MPJPE mm | FK/XYZ不一致 mm | 重建通过 |
|---|---:|---:|---:|---:|
{chr(10).join(quality)}

![成对控制响应](../outputs/OF008-analysis/paired_response.png)

虚线为GT，误差棒为八个采样seed的标准差。角度只用于构造监督和评估，不输入模型。

| Kimodo266主分组 | gain | relative error | 响应通过 |
|---|---:|---:|---:|
{chr(10).join(groups)}

| 次要推理布局 | gain | relative error | 响应通过 |
|---|---:|---:|---:|
{chr(10).join(secondary)}

主gain差（Kimodo266−基线）为 {d['mean']:.8f}，按八个噪声seed整块重采样4000次的95%区间 [{d['ci95'][0]:.8f}, {d['ci95'][1]:.8f}]。该区间仅反映采样噪声；每模型只有一个训练seed，不能用于跨训练种子的显著性结论，也不能把很小的数值差当作实用控制改善。

## 拟合、时间变化与物理一致性

最后1000步已记录的teacher重建损失均值 {a['last_1000_logged_steps']['teacher']['mean']:.6f}；rollout均值 {a['last_1000_logged_steps']['rollout']['mean']:.6f}。下面的时间变化量是在每个关节上排除已知帧后，减去该关节的时间均值，再计算世界位置RMS；汇总主协议80例。这是事后描述性诊断，不能独立定位优化失败的原因。

| 模型 | 输出时间变化RMS mm | GT时间变化RMS mm | 比值 |
|---|---:|---:|---:|
{chr(10).join(temporal)}

![手腕时间轨迹](../outputs/OF008-analysis/temporal_wrist.png)

![训练损失](../outputs/OF008-analysis/training_loss.png)

| 常规80例指标 | 均值 |
|---|---:|
{chr(10).join(phys)}

全部240次生成均通过逐采样步归一化锚点严格相等断言；共同评估物理坐标最大锚点误差 {k['max_physical_anchor_error_m']:.9g} m。精确回填不是自由区域控制成功，也不保证冗余旋转/速度/FK一致。

## CPU控制信息诊断

训练期间额外进行一次无GPU的事后诊断：以布局A已知XYZ为输入，只用三个训练状态拟合带截距的最小二乘映射到完整266D。不输入角度或状态编号，输入中心和尺度也仅由训练状态计算。训练设计矩阵秩为 {probe['training_design_rank']}。

对未训练的−6/+6°，gain分别为 {probe['rows'][1]['common_response_gain']:.6f}/{probe['rows'][3]['common_response_gain']:.6f}，relative error分别为 {probe['rows'][1]['common_response_relative_error']:.6f}/{probe['rows'][3]['common_response_relative_error']:.6f}。这是单动作、固定布局的确定性监督插值，不是生成模型、没有随机噪声，也不验证其它动作或布局。它说明当前A布局的控制信息足以区分该状态族，不能将失败解释为“输入完全不含状态信息”；仍无法排除目标权重、优化、数值精度、时间编码或rollout等环节。

下一步应先隔离拟合/优化问题，例如固定单一去噪格点并比较teacher与rollout训练，检查时间特征和控制分支的逐层敏感度，再决定增加数据或步数。本报告不启动新的付费训练。

## 验证、资源和复现

本地46项CPU测试与远端46项测试通过。源码、官方组件、数据、配置、审批哈希及基线checkpoint均验证。下载后逐文件核验 {r['evidence_files_verified']} 个证据文件；用NumPy独立复算新模型160份共同样本和10份常规样本的XYZ与控制响应，并重审160份基线共同样本。物理旋转/FK指标来自既定评价代码，未宣称所有物理指标均由第二实现复算。240次生成含重复条件，不是240个独立样本。

训练日志耗时 {a['train_seconds']/60:.2f} 分钟，峰值allocated显存 {a['peak_memory_allocated_gb']:.3f} GB。此耗时包含训练入口初始化与周期保存，不是流式推理延迟。

GPU为 {r['gpu']}，${r['rate_usd_hour']:.2f}/小时；Pod `{r['pod_id']}` 已删除，{r['zero_pods_confirmed_at']} 确认零Pod。按创建至零Pod确认时间估算本轮GPU费用上界 **${r['gpu_cost_upper_estimate_usd']:.6f}**；保留网络卷的存储费另计，实际结算以账单为准。

- 已批准方案：`reports/approval_report_v9_zh.md`（原文冻结，描述的是启动前状态）。
- 配置：`configs/kimodo266_overfit_v1.json`。
- checkpoint仅在保留卷：`{r['checkpoint']['path']}`；SHA256 `{r['checkpoint']['sha256']}`。
- 本地证据：`outputs/OF008-evidence.tar.gz`；SHA256 `{r['evidence_sha256']}`。
- 源码包SHA256：`{r['source_archive_sha256']}`。
- 汇总：`outputs/OF008-analysis/analysis.json`，CPU诊断：`outputs/OF008-analysis/cpu_condition_probe.json`。

本轮未运行正式HumanML3D FID/文本匹配、BABEL、全数据训练、噪声锚点或流式延迟测试，不对这些能力作结论。
'''
(root/'reports/OF008_results_zh.md').write_text(text,encoding='utf-8')
print('reports/OF008_results_zh.md')

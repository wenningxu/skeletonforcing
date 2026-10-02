"""Write the OF009 result report from verified local evidence and resource records."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
analysis=json.loads((root/'outputs/OF009-analysis/summary.json').read_text())
resource=json.loads((root/'reports/resources_of009.json').read_text())
rows=json.loads((root/'outputs/OF009/evaluation.json').read_text())
cfg=json.loads((root/'configs/root_first_overfit_v1.json').read_text())
logs=[json.loads(s) for s in (root/'outputs/OF009/train.jsonl').read_text().splitlines()]
mean=analysis['mean'];g=cfg['overfit_gate']
gates=[('feature MSE < 0.01',lambda r:r['feature_mse']<g['feature_mse_max']),
       ('world XYZ MPJPE < 20 mm',lambda r:r['xyz_mpjpe_m']<g['free_xyz_mpjpe_m_max']),
       ('rotation normalized MSE < 0.01',lambda r:r['rot6d_normalized_mse']<g['rotation_normalized_mse_max']),
       ('velocity normalized MSE < 0.01',lambda r:r['velocity_normalized_mse']<g['velocity_normalized_mse_max']),
       ('velocity error / static reference < 0.25',lambda r:r['velocity_error_vs_static_ratio']<g['motion_velocity_error_vs_static_max']),
       ('no degenerate rotations',lambda r:r['degenerate_rotation_count']==0),
       ('FK disagreement < GT floor + 20 mm',lambda r:r['fk_position_disagreement_m']<r['gt_fk_position_disagreement_m']+g['fk_disagreement_above_gt_m_max']),
       ('temporal RMS / GT in [0.8, 1.2]',lambda r:.8<=r['temporal_rms_ratio']<=1.2)]
lines=['# OF009 等步数 root 起点山谷扩散：实验结果','',
       '2026-10-02（Asia/Shanghai）。用户明确要求开始报告 v11 的实验。', '',
       f"**完成一次 4,000 步训练和全部 8 个未用于训练的噪声测试。综合过拟合门槛通过 {analysis['passed']}/8。**",'',
       '## 协议与实际执行','',
       '标准化 Wan `wan_joint_time_v1`，1024宽/8层/8头/FFN2048，78,053,389参数；无文本、无动作VAE、266D冗余表示。训练数据仅动作011936的64帧0°状态，使用已验证OF003状态库既有的FK/速度一致化处理。XYZ统计由该单训练状态拟合，其余200通道沿用官方HumanML3D统计。这不是原生263D全数据训练或FloodDiffusion完整复现。','',
       '起点为第0帧/root0的日程坐标；没有真实root、姿态或控制输入，known mask恒为false，全部266D从高斯噪声开始。沿时间和骨骼链错峰启动相同32步局部线性去噪曲线；root在全局32完成，第0帧腕部51完成，最晚关节帧53完成。总53次网络调用，每个通道恰好32次状态更新；已完成的模型状态冻结并继续作为注意力上下文。当前完整注意力不构成因果流式系统。','',
       '训练batch3为同一动作的独立噪声。AdamW lr2e-4、100步预热、梯度裁剪1、FP32参数/BF16 autocast。前400步teacher；之后以50%概率使用模型实际rollout状态，前缀截断、最后最多4步保留梯度。主目标为当前活动通道的x0 MSE，另加0.01 root/速度一致性项，并屏蔽尚未可用的相邻位置/朝向依赖；没有对off-trajectory状态伪造原epsilon-x速度目标。每53步打乱遍历同一推理完整时间场。','',
       f"实际53个格点各覆盖 {analysis['training_step_coverage_min']}–{analysis['training_step_coverage_max']} 次，合计4,000次。相同采样更新数不代表相同训练梯度权重：仍按当前活动通道平均MSE。",'',
       '## 终态结果','',
       '| 指标 | 八个噪声种子均值 | 范围 |','|---|---:|---:|']
units={'xyz_mpjpe_m':1000,'root_mpjpe_m':1000,'fk_position_disagreement_m':1000,'gt_fk_position_disagreement_m':1000}
labels={'feature_mse':'归一化特征MSE','xyz_mpjpe_m':'world XYZ MPJPE (mm)','root_mpjpe_m':'root MPJPE (mm)',
        'rot6d_normalized_mse':'旋转归一化MSE','velocity_normalized_mse':'速度归一化MSE',
        'velocity_error_vs_static_ratio':'运动速度误差/静止参考','fk_position_disagreement_m':'FK/XYZ不一致 (mm)',
        'gt_fk_position_disagreement_m':'GT FK/XYZ底噪 (mm)','temporal_rms_ratio':'时间变化RMS/GT'}
for k,v in mean.items():
    scale=units.get(k,1);lo,hi=analysis['range'][k]
    lines.append(f'| {labels[k]} | {v*scale:.6g} | {lo*scale:.6g}–{hi*scale:.6g} |')
lines+=['','## 逐项门槛','', '| 门槛 | 通过数 |','|---|---:|']
for label,test in gates:lines.append(f'| {label} | {sum(test(r) for r in rows)}/8 |')
lines+=['', '| 噪声seed | XYZ MPJPE (mm) | 时间RMS/GT | 综合通过 |','|---|---:|---:|---|']
for r in rows:lines.append(f"| {r['seed']} | {r['xyz_mpjpe_m']*1000:.4f} | {r['temporal_rms_ratio']:.6f} | {r['overfit_passed']} |")
lines+=['', '## 波前与训练诊断','', '| 骨骼深度 | 最终XYZ MPJPE (mm) |','|---|---:|']
for k,v in analysis['by_depth'].items():lines.append(f'| {k} | {v*1000:.4f} |')
lines+=['', '| 帧块 | 最终XYZ MPJPE (mm) |','|---|---:|']
for k,v in analysis['by_frame_block'].items():lines.append(f'| {k}–{int(k)+15} | {v*1000:.4f} |')
lines+=['',f"训练最后记录耗时 {logs[-1]['elapsed_s']:.2f} 秒；峰值allocated显存 {max(r['peak_memory_allocated_gb'] for r in logs):.3f} GB。训练日志每25步记录一次，以下均值只对保存日志点计算，不能称为全部训练步精确均值。"]
for flag in [False,True]:
    subset=[r for r in logs if r['step']>3000 and r['rollout']==flag]
    if subset:lines.append(f"最后1,000步保存的 {'rollout' if flag else 'teacher'} 点：{len(subset)}条，平均活动重建MSE {sum(r['reconstruction'] for r in subset)/len(subset):.6g}。格点/活动通道不同，不能把两个分支均值作为同状态因果对照。")
lines+=['', '![训练与终态诊断](../outputs/OF009-analysis/diagnostics.png)','',
        '[GT与最终生成动作并排动画](../outputs/OF009-analysis/reference_vs_generated.gif)。动画展示最终生成序列随动作帧播放，不是去噪波前动画。','',
        '## 可得结论与边界','',
        '八项逐项指标中六项全部通过，失败的两项为XYZ位置精度和FK一致性。时间RMS接近GT且速度误差低于静止参考的0.25门槛，说明本次没有出现此前那种明显时间幅度塌缩；这仍不能代替精确重建或物理一致性。root和深度6/7末端位置误差较大，但这些结果不能单独证明错峰冻结、训练步数或某个组件是原因。', '',
        '本次只检验一个动作、一个训练状态、一个训练seed的无条件过拟合。独立测试噪声不等于新动作/新布局泛化。没有用户控制输入，不报告锚点精度或控制响应。没有同计算量的均匀时间场对照，不能证明山谷日程优于均匀扩散；没有FID、正式评估器全数据成绩或流式延迟测试。误差与物理一致性分别报告，不能用训练loss或某一项通过替代综合门槛。','',
        '## 证据与资源','',
        '56项本地与远端环境测试通过。下载包和逐文件SHA256核验；8个保存样本由CPU独立复核XYZ/归一化误差、FK/速度、时间RMS及综合门槛。保存快照中已完成通道逐位冻结，远端逐步检查所有未活动通道不变；时间场各通道32次更新由本地重新计算核验。','',
        '执行准备出现三次训练前错误：shell CRLF、历史Kimodo测试缺pydantic、上传包漏两份审计工具。均在任何训练状态创建前停止，修正打包/依赖后才运行一次真正训练；模型、配置及批准报告未变。第一台4500在上传阶段取消并删除，未训练；随后使用4090。启动PID及记录见resources_of009.json，所有setup日志随证据保存。','',
        f"4090 Pod `{resource['pod_id']}` 状态 **{resource['status']}**；GPU报价 $0.74/小时，90分钟/$2上限。GPU成本上界估计 ${resource.get('gpu_cost_upper_estimate_usd',0):.6f}；账单未核定，存储费另计。取消的4500分配费用单列于of009_cancelled_allocation.json。原100GB卷k1j34iuc97保留。",'',
        f"完整checkpoint仅在卷上：`{resource['checkpoint']['path']}`，SHA256 `{resource['checkpoint']['sha256']}`。",'',
        f"本地证据：`outputs/OF009-evidence.tar.gz`，SHA256 `{resource['evidence_sha256']}`；核验{resource['evidence_files_verified']}份证据文件。完整checkpoint没有下载或提交GitHub。",'',
        '详见outputs/OF009、outputs/OF009-analysis/summary.json及reports/resources_of009.json。此次单次运行授权已使用，不得重启/重跑OF009。后续有界Wan测试仍应先形成新的具体方案，扩大模型/数据/噪声锚点范围不在本轮授权内。','']
(root/'reports/OF009_results_zh.md').write_text('\n'.join(lines),encoding='utf-8')
print(root/'reports/OF009_results_zh.md')

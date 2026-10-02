# 去噪网络组件标准化 v1：已实现，尚未进行GPU实验

2026-09-25。本轮按用户要求先标准化网络组件，再讨论训练效果。新增 `WanMotion266`，使用与 Flood/Wan 对齐的去噪 block、时间调制和输出头。旧OF001–OF003代码与结果仍作为历史基线，不能将新结构描述为那些实验实际使用的结构。

![已实现结构](../outputs/architecture/wan_standard_v1.png)

[可缩放SVG](../outputs/architecture/wan_standard_v1.svg)

## 1. 对齐到什么程度

标准组件移植在 [wan_components.py](../motion_valley/wan_components.py)，来源是已保存的官方 `third_party/FloodDiffusion/models/tools/wan_model.py`，保留原版权声明和 Apache-2.0 来源说明。不是仅借用“DiT”名称重写普通Transformer。

每个block严格采用以下顺序：

1. 无仿射 LayerNorm，FP32归一化；由时间条件产生shift/scale，输入双向Self-Attention，随后使用时间gate控制残差。
2. 带仿射LayerNorm → 文本Cross-Attention → 残差。
3. 无仿射LayerNorm，时间shift/scale → Linear/GELU(tanh)/Linear FFN → 时间gate残差。

Q/K在划分head之前沿完整投影宽度做RMSNorm，位置使用与Wan一致的复数RoPE及三轴频率分配。调制来自共享时间MLP；每层有独立的可学习modulation偏置，不是每层再建立一套时间MLP。输出头同样使用时间shift/scale，而不是普通LayerNorm后直接输出。

初始化沿用发布代码的主要规则：Linear的Xavier初始化、文本和时间embedding权重标准差0.02、随机modulation偏置、最终输出Linear权重为0。**不把它误称为“所有block的AdaLN-Zero门均为零”**，原代码并非如此。

为了本地CPU验证及后续GPU可移植性，注意力后端使用PyTorch SDPA，而非要求安装官方FlashAttention CUDA扩展。注意力与block数学公式经过独立对照；尚未证明两个GPU kernel的低精度舍入和性能完全一致。

## 2. 必要的动作空间适配

动作仍为266D：root4 + worldXYZ66 + rot6d126 + local velocity66 + contact4。没有运动VAE，不改变已批准的干净锚点原则。

`B×T×266 → B×T×22×13 → B×(T·22)×D`。13是按关节打包后的槽宽，补零槽位有显式live mask，不是13维动作潜空间。motion投影、已知通道mask投影和关节embedding形成初始token；不再在这里加入平均σ的时间embedding。

本版**取消旧模型“空间Transformer再时间Transformer”的轴向拆分**，每个标准Wan block在完整joint×time token集合上做双向自注意力。这样减少额外骨干变体：解耦发生在token布局和时间场，不要求注意力也必须轴向分解。该变化意味着它不是“旧网络只加AdaLN”的单因素消融；本版的任务是建立标准结构起点。

RoPE坐标为 `(frame_index, joint_index, 0)`，沿用Wan三轴公式。关节编号只表示固定身份顺序，不等价于骨架图距离；骨架图距离仍用于谷底时间场。加入此位置轴是显式动作适配，不称为原版Flood的latent时间轴设置。

完整自注意力的代价随 `(T·22)²` 增长，64帧有1408个token。相比原latent网络或轴向实现，算力/显存开销都不同；后续正式实验必须在批准后先检查资源占用，不能沿用旧模型的运行时间估计或声称已具备实时性。

## 3. 时间场如何进入每一层

标准标量时间输入必须适配为每个关节token的完整通道噪声向量。做法为：

`σ[B,T,266] → pack → σ[B,T,22,13]`

`每个通道分别sinusoidal embedding → live mask去除补零槽 → 按通道顺序拼接`

`共享 Linear(13·freq_dim,D) → SiLU → Linear(D,D) → e[B,T·22,D]`

`e → SiLU → Linear(D,6D) → modulation[B,T·22,6,D]`

每个block把共享modulation加上本层偏置，得到Self-Attention和FFN两组shift/scale/gate。各关节和帧保留不同条件，**没有全动作池化，也没有用平均σ替代完整通道时间信息**。同一关节的干净XYZ与仍带噪的旋转/速度可以被区分。

时间embedding采用与官方一致的cos/sin频率顺序，scale=1.0；时间MLP、调制残差和输出头使用FP32，注意力/FFN可在autocast下使用BF16。权重保持FP32，不要求把整个模型强制转换为BF16。

新增模型保留原采样接口。由 `flow_features.py` 生成准确完整时间场，仍按x0预测更新自由通道，再投影已知坐标；σ非零的已知通道会被模型拒绝。x0目标保留用于模型rollout监督；没有未经讨论改回对离轨状态不成立的原始velocity标签。网络输出头本身不硬拷贝控制点，精确约束由采样/训练的同一控制投影负责。

## 4. 文本与尺寸配方

文本分支保持4096D官方UMT5 token接口，使用Linear/GELU/Linear投影。每层Cross-Attention支持padding mask及逐帧允许的文本mask；无文本配置物理移除文本投影、cross-attention和对应norm，不接收dummy token。两配置在相同种子下先共同初始化再删除文本模块，保留共同运动参数的一致性。

| 配方 | D | block | head | FFN | 生成器参数，不含冻结UMT5 |
|---|---:|---:|---:|---:|---:|
| 主模型尺寸 | 1024 | 8 | 8 | 2048 | 116,918,285 |
| 诊断尺寸，有文本 | 256 | 8 | 8 | 1024 | 10,883,597 |
| 诊断尺寸，无文本 | 256 | 8 | 8 | 1024 | 7,655,437 |

主模型尺寸在width/depth/head/FFN上对应Flood主配置，但输入、时间适配器与输出仍是本项目动作空间版本，参数量无需与原latent模型相等。诊断配方明确是较小尺寸，不冒充原主模型规模；两者使用完全相同的组件代码。

配方文件：[主尺寸](../configs/model_wan266_main_v1.json)、[诊断尺寸](../configs/model_wan266_v1.json)。它们只有模型参数，没有steps、预算、数据选择或审批凭据，不是可直接启动的GPU实验配置。

## 5. 验证证据

本地CPU全量31项测试通过，包括6项新增标准组件测试：

- 从实际官方源文件提取**未改动的类定义**作为数值参考，仅以独立softmax/matmul替换无法在CPU使用的FlashAttention后端；复制相同权重，比较block的有效输出、输入梯度、时间条件梯度、文本梯度和全部参数梯度。输出/梯度容差为atol 3e-6，rtol最高4e-5。输出头也单独核对。
- Wan三轴RoPE及sinusoidal编码与官方实现逐位一致。
- 交换同一关节XYZ与旋转通道的σ、保持平均σ不变，时间编码仍发生变化；确认不是平均值条件。
- 每层独立捕获到时间条件的非零梯度，自由输出到干净观察输入存在梯度路径。这是计算图连通性证据，不是已训练的控制效果证据。
- padding动作/文本中的大值及NaN被屏蔽；不允许的文本不进入该帧cross-attention；有效帧完全无文本的非法mask被拒绝。
- CPU BF16下完成两次合成数据优化器更新，随后跑真实采样/rollout代码，验证初始化及每一步干净锚点逐位不变、梯度有限、训练时间场与采样一致。这只是集成冒烟测试，不是动作过拟合实验。

诊断尺寸另用一条本地真实64帧动作检查完整前向，8层输入均为 `[1,1408,256]`，调制均为 `[1,1408,6,256]` 且dtype为FP32，输出 `[1,64,266]` 有限。该检查使用合成文本token，只验证维度接线；零输出符合官方零初始化输出头，不能当作学习结果。主尺寸仅在meta device构建并统计参数，没有执行前向或训练。

运行官方CPU参考时出现4条“CUDA不可用，关闭autocast”的预期警告，未执行CUDA计算，不是失败。GPU低精度对照、实际显存/速度以及真实学习仍待审批后的实验确认。

证据：[检查JSON](local_checks_wan_v1.json)、[测试代码](../tests/test_wan_standard.py)、[真实输入检查脚本](../scripts/check_standard_model.py)。

## 6. 接入和下一步边界

新模型通过显式 `architecture=wan_joint_time_v1` 接入 `control_experiment.py`。旧配置仍明确落到legacy路径；未改变OF003配置与审批报告哈希，不能把旧结果混写成新网络结果。未知architecture名称直接报错。

这一步已经完成本地实现与组件验证。**没有创建/启动GPU，没有进行新一轮动作训练，也不从单元测试推出任何性能改进。** 下一步应围绕标准主干重新制定过拟合方案，固定模型尺寸、数据、条件、损失、时间场、评价门槛和资源上限，提交详细报告后等待用户批准。

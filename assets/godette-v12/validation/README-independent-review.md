# Godette v12（74 mesh）限定独立复验

## 结论

本冻结版通过本报告列出的原件保护、六状态裤体自身与裤踝检测、三条路径共63个离散采样、以及新鞋底封闭性检查。原失败版的后裤腿露皮和鞋底空壳露皮已在相同压力后视与直接仰视渲染中消失。

本结论**不是全模型无相交、皮肤一体无缝或任意动作均通过**。生成踝/脚背接缝、鞋的重叠装配接缝及未覆盖范围见下文。

## 文件与只读保证

- v11源件 SHA-256：`b03224d8683604848b1a241c4a2178bc669124abb1ae005bc92b99495b0b0f77`
- 失败72 mesh候选 SHA-256：`096f4508f342d0366cdb0a0d86a3cd1d485bab8a9290e2787ccad581b224c7d4`
- 通过限定复验的74 mesh候选 `godette-v12-final-candidate.blend` SHA-256：`38666f464943ce2e5e9498ce6d461130e689aafce1c92304ea91a568de0420e4`
- 对应 `v12-final-review.blend` SHA-256：`432a7e6cfcc13e98205f80d1cab065420fdc43f7a12b6b5e8a7015fa11f06ac0`
- 独立使用 Blender 4.3.2，`--disable-autoexec`，所有载入另设 `use_scripts=False`。只在内存变更测试姿态/相机，没有保存改写任何输入blend。前后文件哈希复核结果见 `input-sha256-verification.json`。

## 变化范围与原件保护

相对v11的72件mesh，无删除；仅新增 `Closed outsole plate -1` 与 `Closed outsole plate 1`，共74件。

既有mesh中，仅以下七件坐标变化：一体长裤、左右 Dark cloth flat、左右 Flat thin sole、左右 Clean ankle。Clean ankle另外改变权重并新增相应小腿vertex group。所有既有mesh拓扑、UV、材质赋值、对象变换与修改器配置保持相同；材质设置及节点定义比较无差异。

相对失败72件版，所有既有mesh坐标保持完全相同；仅两件Clean ankle的权重及vertex groups改变，另新增两鞋底。未删除或改写Clean instep。

- 227根骨骼的名称、父子关系、rest数据、骨架对象变换均与v11相同。
- 默认pose只有指定16个Lip骨的location及其对应matrix_basis变化，与neutral preset一致；其他pose通道无变化。
- Object_7全部4815点的原始坐标、拓扑、UV、权重、材质赋值等数据与v11完全相同。
- 手及保留前臂的材质1顶点共2530点，native默认姿态评估坐标零差。2530是手与保留前臂的合计，不把它误称为纯手点数。
- Object_7在native评估后有211点因有意Lip位置调整而变化；不把这误报为原face mesh修改，也不以“全部默认pose零差”否定有意中性化。表情观感与GLB口型另由对应报告验证。

完整结果见 `independent-v11-v12-delta.json` 与 `independent-failed-to-final-additions.json`。

## 六个静态状态

1. Rest空间。
2. Native默认姿态，保留原默认肢体姿态与本次16Lip中性preset。
3. 独立保存的review站姿。
4. 左髋世界X −35°、膝 +80°、足 −15°。
5. 右髋世界X −35°、膝 +80°、足 −15°。
6. 双髋世界X −25°、双膝 +50°、双足 −15°。

每个压力状态都从同一个已保存review站姿分别恢复后执行；顺序为髋、膝、足，以当前骨头位置为旋转中心。Review不是native默认姿态。

六状态的裤体自身非邻接三角候选均为0。裤体对左右踝、鞋面、旧薄底边、新鞋底板的交叉候选也均为0。修前左/右后裤腿分别75/79组三角交线，在新版指定状态中已不再出现。

方法：评估世界坐标三角网格，BVH epsilon=0；裤体自检排除共享任意顶点的邻接三角。若存在候选，则用双向边段—三角求交，交段长度大于1e-6米记为明确交线；接触、共面及未确认项单独记录。

简明结果见 `independent-six-state-summary.json`；逐交线完整记录见 `independent-six-state-deformation.json`。

## 三路径、每条21点

从review站姿线性递增上述左、右、双腿世界轴角度，fraction为0、0.05、…、1.00。63个采样均满足：

- 裤体自身非邻接三角候选0。
- 裤体与左右Clean ankle候选0。
- 评估坐标全部finite。

这是一组离散采样，不是连续碰撞证明，也不覆盖更大屈曲、扭转、内外翻或任意动画。

结果见 `independent-three-path-sweep.json`。

## 新鞋底封闭与仰视

每件Closed outsole plate均为128点、192边、66面，单连通、Euler=2；所有边均恰连两面，边绕向一致，正有向体积。世界Z厚度约4.4999998毫米，全部Foot权重100%。两件板自身非邻接三角候选0。

在rest空间从每个Clean instep三角中心正下方投射射线，共1056条，第一命中全部为对应的新鞋底板；皮肤第一命中与未命中均为0。这是三角中心有限采样，不扩张成所有视角结论。

实际独立Cycles渲染（48 samples，禁用denoise）确认：

- `left_back.png`、`right_back.png`：压力姿态后裤腿不再出现椭圆露皮；抬脚鞋底不再露出脚背皮肤。
- `both_back.png`：双腿压力姿态同样通过上述视觉检查。
- `standing_underside.png`：从鞋底正下方观察，左右均为封闭底板。
- `standing_back.png`、`left_side.png`：补充后裤口和侧面接触外观。

数值见 `independent-sole-underside-coverage.json`。

## 明确保留的接触和局限

- 鞋面、旧薄底边和新鞋底板是重叠装配，并未布尔合并为整鞋单一无交叉壳体。报告保留这些真实交线，不能宣称鞋的所有对象之间无相交。
- Clean ankle与Clean instep仍是两块相交皮肤。左右各47组三角交线，在修前/修后的review站姿及三压力状态计数和四向采样可见性相同。压力侧视仍能看到该皮肤接缝。这不是本次权重修复新增问题，也不应写成“皮肤一体无缝”。见 `supplemental-instep-seam-audit.json`。
- 原手壳体缺陷没有在本次修复。外套/裤腰既有接触没有在本次重新作完整相交审计。
- 没有证明全模型所有对象对、任意镜位或全部动作无穿插。
- 脸部表情评价、导出GLB及其往返一致性不在本独立变形报告的结论范围。

## 交付入口

- `independent-final-verdict.json`：简明限定结论。
- `independent-six-state-summary.json`：六状态小体积摘要。
- `independent-three-path-sweep.json`：63个采样。
- `independent-v11-v12-delta.json`：v11到最终版差异。
- `independent-failed-to-final-additions.json`：失败版到最终版差异、鞋底拓扑与踝权重函数校验。
- `independent-sole-underside-coverage.json`：鞋底自检与1056条仰视射线。
- `supplemental-instep-seam-audit.json`：保留皮肤接缝的修前/修后对比。
- 六张PNG：独立视觉复核。
- `manifest.json`：本目录报告、脚本及图像哈希。

失败版证据仍保留在上一级目录，没有用新版结果覆盖旧失败记录。

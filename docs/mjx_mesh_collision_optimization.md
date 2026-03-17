# MJX Mesh 碰撞优化排查记录

> XLEO 双手按摩任务 JIT 编译极慢 & 显存爆炸的根因分析与修复

## 1. 问题现象

XLEO 双手按摩任务（`massage.py`）的 JIT 编译时间异常：

| 任务 | JIT 编译时间 |
|------|-------------|
| **XLEO Massage** | **~3.5 分钟** |
| LEAP Reorient（同量级） | ~70 秒 |

两个任务的 `njmax`/`naconmax` 设置相同，模型自由度也相近，但编译时间相差 3 倍。

## 2. 初始排查方向（碰壁）

最初怀疑是 `njmax`/`naconmax` 设置问题，但检查后发现两个任务设置一致：

| 参数 | 值 |
|------|-----|
| `naconmax` | 30 × 8192 = 245,760 |
| `njmax` | 160 |

这是一个**误导方向**——这些参数控制的是接触缓冲区大小，不是编译时间的瓶颈。

## 3. 模型维度对比

通过 Python 脚本对比两个任务的模型结构，找到了关键差异：

```python
import mujoco
from etils import epath
from collections import Counter

# --- XLEO Massage ---
from mujoco_playground._src.manipulation.xleo_hand.massage import get_assets as xleo_assets
from mujoco_playground._src.manipulation.xleo_hand import constants as xleo_consts
xleo_model = mujoco.MjModel.from_xml_string(
    epath.Path(xleo_consts.SCENE_XML.as_posix()).read_text(), assets=xleo_assets()
)

# --- LEAP Reorient ---
from mujoco_playground._src.manipulation.leap_hand_study.reorient import get_assets as leap_assets
from mujoco_playground._src.manipulation.leap_hand_study import leap_hand_constants as leap_consts
leap_model = mujoco.MjModel.from_xml_string(
    epath.Path(leap_consts.CUBE_XML.as_posix()).read_text(), assets=leap_assets()
)

print('=== Model Dimensions ===')
for name, m in [('XLEO Massage', xleo_model), ('LEAP Reorient', leap_model)]:
    print(f'\n{name}:')
    print(f'  nq={m.nq}, nv={m.nv}, nu={m.nu}')
    print(f'  nbody={m.nbody}, ngeom={m.ngeom}, nmesh={m.nmesh}')
    print(f'  njnt={m.njnt}')
    col_geoms = [i for i in range(m.ngeom) if m.geom_contype[i] > 0]
    mesh_col = [i for i in col_geoms if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
    prim_col = [i for i in col_geoms if m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH]
    print(f'  collision geoms: {len(col_geoms)} (mesh: {len(mesh_col)}, primitive: {len(prim_col)})')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f'  ncon (contacts at qpos0): {d.ncon}')
    type_names = {0:'plane',1:'hfield',2:'sphere',3:'capsule',4:'ellipsoid',5:'cylinder',6:'box',7:'mesh'}
    types = Counter(type_names.get(m.geom_type[i], f'type{m.geom_type[i]}') for i in col_geoms)
    print(f'  collision geom types: {dict(types)}')
```

**输出结果：**

```
=== Model Dimensions ===

XLEO Massage:
  nq=30, nv=30, nu=30
  nbody=22, ngeom=42, nmesh=22
  njnt=30
  collision geoms: 22 (mesh: 20, primitive: 2)
  contact excludes (npair exclude): 12
  ncon (contacts at qpos0): 2
  collision geom types: {'plane': 1, 'mesh': 20, 'capsule': 1}

LEAP Reorient:
  nq=23, nv=22, nu=16
  nbody=21, ngeom=61, nmesh=12
  njnt=17
  collision geoms: 37 (mesh: 0, primitive: 37)
  contact excludes (npair exclude): 30
  ncon (contacts at qpos0): 0
  collision geom types: {'plane': 1, 'box': 36}
```

**关键差异一目了然：**

| | XLEO Massage | LEAP Reorient |
|---|---|---|
| 碰撞 geom 数 | 22 | 37 |
| **mesh 碰撞体** | **20** | **0** |
| primitive 碰撞体 | 2 (plane + capsule) | 37 (plane + box) |
| contact excludes | 12 | 30 |

LEAP 虽然碰撞体更多（37 vs 22），但全部是 **box primitive**，没有 mesh；XLEO 有 **20 个 mesh 碰撞体**。

## 4. MJX vs MuJoCo CPU 碰撞检测原理差异

这是理解问题的核心。

### MuJoCo CPU（运行时动态）

- **Broadphase**：用 AABB 包围盒快速排除不可能碰撞的 geom 对
- 只有少数空间上接近的 geom 对进入 **narrowphase** 做精确碰撞检测
- 每帧实际碰撞对很少（观测到 `ncon=2`）

### MJX（编译时静态）

- JIT 编译时必须将**所有可能的碰撞对**写入 XLA 计算图
- JAX 计算图形状固定，**不能动态跳过**——没有数据依赖的分支裁剪
- 每个潜在碰撞对都会生成碰撞检测代码，**即使 99% 在运行时永远不会接触**

### 不同碰撞类型的代价差异

| 碰撞类型 | 算法 | XLA 计算图大小 |
|----------|------|---------------|
| box-box / capsule-box | 解析公式 | 极小 |
| mesh-primitive | GJK（简化） | 中等 |
| **mesh-mesh** | **GJK + EPA（完整）** | **极大** |

mesh-mesh 的 GJK/EPA 是迭代算法——MJX 必须通过 `lax.while_loop` 展开，每次迭代遍历所有顶点计算 support function。计算图大小与顶点数成正比。

## 5. Mesh-Mesh 碰撞对数爆炸

20 个 mesh geom 在默认 `contype=1 conaffinity=1` 下，任意两个都会碰撞：

$$C(20, 2) = \frac{20!}{2! \cdot 18!} = 190 \text{ 对 mesh-mesh 碰撞}$$

减去 12 个 contact exclude，仍有 **~178 对 mesh-mesh** 碰撞检测需要编译进 XLA 图。

**代价估算：**

```
修复前: 178+ 对 × mesh-mesh GJK/EPA = 巨大的计算图 → 3.5 分钟 JIT + 显存爆炸
修复后:  ~40 对 × mesh-primitive GJK = 小计算图 → 正常 JIT 时间
```

**显存方面**：XLA 计算图 → 更大的 HLO 编译产物 → 运行时更多中间缓冲区。190 对 mesh-mesh 的中间张量直接撑爆显存。

## 6. 参考方案

从项目中另一个手部模型 `xleohand_lh_wrist_msg.xml` 中发现了正确的碰撞配置做法：

```xml
<!-- xleohand_lh_wrist_msg.xml 中的碰撞配置 -->
<geom contype="1" conaffinity="0" group="3"/>  <!-- link_geom class：手指 mesh -->
```

| class | contype | conaffinity | 效果 |
|-------|---------|-------------|------|
| `link_geom`（mesh） | 1 | **0** | 手指 mesh 之间不互相碰撞 |
| `tactile`（sphere） | 2 | **0** | 触觉球不和任何东西碰撞 |
| `convex_decomposition`（人体 capsule） | 1 | **15** (=0b1111) | 和 contype 1/2/4/8 的物体都碰撞 |

## 7. 最终修复

**文件**：`mujoco_playground/_src/manipulation/xleo_hand/models/xmls/ftl_xleo_dual_hand.xml`

**一行修改：**

```diff
 <default class="collision">
-  <geom type="mesh" group="3" rgba="1 0.5 0 0.5"/>
+  <geom type="mesh" group="3" rgba="1 0.5 0 0.5" contype="1" conaffinity="0"/>
 </default>
```

**修复后碰撞检查验证：**

```
collision geoms: 22
  floor:       contype=1 conaffinity=1
  (×20 mesh):  contype=1 conaffinity=0   ← 手指 mesh 之间不再互碰
  simple_arm:  contype=1 conaffinity=15
ncon at qpos0: 2
```

## 8. MuJoCo 碰撞过滤规则

MuJoCo 判断两个 geom 是否碰撞的核心规则——**位运算**：

```
collide = (contype_A & conaffinity_B) || (contype_B & conaffinity_A)
```

修复后各 geom 对的碰撞判定：

| Geom A | Geom B | contype_A & conaffinity_B | contype_B & conaffinity_A | 碰撞？ |
|--------|--------|--------------------------|--------------------------|--------|
| 手指 mesh (1/0) | 手指 mesh (1/0) | `1 & 0 = 0` | `1 & 0 = 0` | ❌ 不碰撞 |
| 手指 mesh (1/0) | 人体 capsule (1/15) | `1 & 15 = 1` | `1 & 0 = 0` | ✅ 碰撞 |
| 手指 mesh (1/0) | 地板 (1/1) | `1 & 1 = 1` | `1 & 0 = 0` | ✅ 碰撞 |

**完美实现**：手指间不碰撞（消除 190 对 mesh-mesh），但手指仍然能与人体手臂和地板碰撞。

## 9. 遗留问题：双手防穿透

修复后手指 mesh 间的自碰撞被完全禁用，包括**左手和右手之间**也不会碰撞。在双手按摩任务中，两手可能穿透彼此。

### 可选方案

1. **利用 contype/conaffinity 位掩码分组**
   - 左手 mesh：`contype=2 conaffinity=0`
   - 右手 mesh：`contype=4 conaffinity=0`
   - 人体 capsule：`conaffinity=0b0110`（= 6），碰撞 contype 2 和 4
   - 左手 L2 指尖：`contype=2 conaffinity=4`（碰撞右手 mesh）
   - 右手 L2 指尖：`contype=4 conaffinity=2`（碰撞左手 mesh）
   - 效果：仅指尖间跨手碰撞，同手内部不碰撞

2. **用 primitive 代替 mesh 做跨手碰撞**
   - 在每只手的关键位置添加几个 capsule/sphere primitive
   - 这些 primitive 设为 `contype=X conaffinity=Y` 只与对侧手碰撞
   - 保持 mesh 碰撞（手↔人体）不变
   - 优点：primitive 碰撞代价极低，不影响 JIT 时间

3. **奖励函数软约束**
   - 不依赖物理碰撞，在 reward 中惩罚双手过于接近
   - 优点：零碰撞开销
   - 缺点：软约束，不能物理阻止穿透

## 附：碰撞 mesh 简化历史

原始碰撞凸包面片数为 1000~2570 面，在 24GB GPU 上直接 OOM。已简化至 `ftl_meshes/convex_new/` 目录下的 64~128 面版本（11 个独立碰撞 mesh，右手复用左手指节 mesh）。即便如此，190 对 mesh-mesh 仍然不可接受。

---

**总结**：MJX 下 mesh-mesh 碰撞对数量是 JIT 编译时间和显存占用的决定性因素。通过 `contype`/`conaffinity` 位掩码精确控制碰撞过滤，可以在不损失必要碰撞检测的前提下，将编译代价降低数倍。

# MuJoCo 碰撞掩码优化排查记录
> 地板 geom 碰撞属性未正确配置，导致训练速度下降约 10 倍的根因分析与修复
## 1. 问题现象

在 `ftl_xleo_dual_hand.scene.xml` 场景中，训练速度异常缓慢。经过排查发现，仅修改地板 geom 的碰撞属性，训练速度即提升约 **10 倍**。
```xml
<!-- 修改前 -->
<geom name="floor" size="0 0 0.05" type="plane" material="groundplane" group="1"/>
<!-- 修改后 -->
<geom name="floor" pos="0 0 -0.25" size="0 0 0.01" type="plane" material="groundplane"
      contype="2" conaffinity="2"/>
```

问题是：地板和手部 mesh 在物理上并不接触（手在空中操作），为什么仅仅改变碰撞属性就能带来如此大的性能差异？
## 2. 根因分析：碰撞检测的隐性开销
### MuJoCo 碰撞检测流程

MuJoCo 的碰撞检测分三个阶段：
```
宽相位 (Broadphase)  →  窄相位 (Narrowphase)  →  接触求解 (Contact Solve)
      ↑                        ↑                        ↑
   每步必跑              有候选对就执行             有接触点才执行
```
- **宽相位**：先检查 `contype/conaffinity` 位掩码，不匹配则直接跳过；匹配则继续检查 AABB 包围盒是否重叠
- **窄相位**：对 AABB 重叠的 geom 对执行精确几何碰撞计算（如 mesh vs plane）
- **接触求解**：对窄相位确认接触的 geom 对计算接触力
### 问题出在哪里

修改前，地板未指定 `contype`/`conaffinity`，使用默认值 `contype=1, conaffinity=1`。手部碰撞体设置为 `contype=1, conaffinity=0`。

MuJoCo 判断两个 geom 是否需要碰撞检测的规则是**位运算**：
```
碰撞条件 = (A.contype & B.conaffinity) || (B.contype & A.conaffinity) != 0
```

修改前，地板与手部 mesh 的碰撞判定：
```
手部.contype(01) & 地板.conaffinity(01) = 01 = 1  ✅
地板.contype(01) & 手部.conaffinity(00) = 00 = 0  ❌

1 || 0 = 1 → 需要碰撞检测
```

位掩码匹配通过后，进入 AABB 包围盒检查。但 `plane`（无限平面）的 AABB 是**无限大的**，所有手部 geom 的 AABB 都与之重叠，因此**每一对都进入窄相位**。

窄相位需要执行 **mesh vs plane** 的精确几何计算。双手模型有几十个碰撞 mesh geom，**每个 timestep 都要做几十次 mesh-plane 精确距离计算**。即使最终没有实际接触点，这些计算本身就非常昂贵。
**核心问题**：检测"没有碰撞"本身也在消耗大量算力。
## 3. 修复方案
### 方案原理

只要让地板与手部 mesh 的位掩码不匹配，就能在宽相位最早期将这些碰撞对过滤掉——位掩码检测只需一次整数位运算，几乎零成本。
### 位掩码的频道直觉

可以把每个 bit 位理解为一个"频道"：
| 值 | 二进制 | 频道 |
|----|--------|------|
| 1  | `01`   | 频道 0 |
| 2  | `10`   | 频道 1 |
| 3  | `11`   | 频道 0 和 1 |

只有两个 geom 在同一频道上才会碰撞。手部在频道 0，将地板改到频道 1，即可互不干扰。
### 具体修改

将地板设置为 `contype=2, conaffinity=2`（二进制 `10`）：
```
手部.contype(01) & 地板.conaffinity(10) = 00 = 0  ❌
地板.contype(10) & 手部.conaffinity(00) = 00 = 0  ❌

0 || 0 = 0 → 不产生碰撞检测，直接跳过
```

几十个碰撞对在最早期就被全部过滤，连 AABB 都不需要计算。
### 其他等效写法

只要让位掩码公式结果为 0 即可。手部为 `contype=1, conaffinity=0` 时：
| 地板设置 | 计算 | 结果 |
|---------|------|------|
| `contype=2, conaffinity=2` | `(1&2)\|\|(2&0)=0` | ✅ 不碰撞 |
| `contype=2, conaffinity=0` | `(1&0)\|\|(2&0)=0` | ✅ 不碰撞 |
| `contype=0, conaffinity=0` | `(1&0)\|\|(0&0)=0` | ✅ 不碰撞 |
| `contype=0, conaffinity=2` | `(1&2)\|\|(0&0)=0` | ✅ 不碰撞 |

最简单的写法：
```xml
<geom name="floor" ... contype="0" conaffinity="0"/>
```

也可以使用 `<exclude>` 标签逐对排除，但对于多 body 模型较繁琐：
```xml
<contact>
  <exclude body1="world" body2="LINK_HAND_BASE_L"/>
  <exclude body1="world" body2="LINK_HAND_BASE_R"/>
  <!-- ... -->
</contact>
```
## 4. 总结
| | 修改前 | 修改后 |
|---|---|---|
| 地板碰撞属性 | 默认 `contype=1, conaffinity=1` | `contype=2, conaffinity=2` |
| 与手部 mesh 位掩码匹配 | ✅ 匹配，进入后续检测 | ❌ 不匹配，直接跳过 |
| 每步窄相位计算 | 几十次 mesh-plane 精确计算 | 0 次 |
| 训练速度 | 基准 | **~10 倍提升** |
**经验教训**：在仿真场景中，应当通过 `contype`/`conaffinity` 位掩码在最早期过滤掉不需要的碰撞对。尤其是涉及无限平面（plane）和大量碰撞 mesh 的场景——plane 的 AABB 无限大，会导致所有 geom 都通过宽相位进入昂贵的窄相位计算，即使它们在物理上根本不会接触。

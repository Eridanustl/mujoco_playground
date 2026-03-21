# MuJoCo 碰撞掩码优化：训练速度提升 10 倍的原因分析

## 背景

在 `ftl_xleo_dual_hand.scene.xml` 中对地板 geom 的碰撞属性做了一处修改，训练速度提升了约 10 倍。

## 关键改动

```xml
<!-- 改动前 -->
<geom name="floor" size="0 0 0.05" type="plane" material="groundplane" group="1"/>

<!-- 改动后 -->
<geom name="floor" pos="0 0 -0.25" size="0 0 0.01" type="plane" material="groundplane"
      contype="2" conaffinity="2"/>
```

## MuJoCo 碰撞检测机制

MuJoCo 通过位掩码（bitmask）判断两个 geom 是否需要碰撞检测：

```
碰撞条件 = (A.contype & B.conaffinity) || (B.contype & A.conaffinity) != 0
```

这里使用的是**按位与（bitwise AND）**运算。

### 改动前

地板未指定 `contype`/`conaffinity`，默认值均为 **1**（二进制 `01`）。
手部碰撞体设置为 `contype=1, conaffinity=0`：

```
手部.contype(01) & 地板.conaffinity(01) = 01 = 1  ✅
地板.contype(01) & 手部.conaffinity(00) = 00 = 0  ❌

1 || 0 = 1 → 产生碰撞检测
```

### 改动后

地板设置为 `contype=2, conaffinity=2`（二进制 `10`）：

```
手部.contype(01) & 地板.conaffinity(10) = 00 = 0  ❌
地板.contype(10) & 手部.conaffinity(00) = 00 = 0  ❌

0 || 0 = 0 → 不产生碰撞检测
```

### 位掩码的频道直觉

可以把每个 bit 位理解为一个"频道"：

| 值 | 二进制 | 频道 |
|----|--------|------|
| 1  | `01`   | 频道 0 |
| 2  | `10`   | 频道 1 |
| 3  | `11`   | 频道 0 和 1 |

只有两个 geom 在同一频道上才会碰撞。手部在频道 0，改动后地板在频道 1，互不干扰。

## 为什么即使不实际碰撞也会慢？

MuJoCo 的碰撞检测分三个阶段：

```
宽相位 (Broadphase)  →  窄相位 (Narrowphase)  →  接触求解 (Contact Solve)
      ↑                        ↑                        ↑
   每步必跑              有候选对就执行             有接触点才执行
```

### 1. 宽相位（Broadphase）

每个 timestep 检查哪些 geom 对需要碰撞检测：

1. 先看 `contype/conaffinity` 位掩码 → 不匹配则**直接跳过**
2. 匹配则继续检查 AABB 包围盒是否重叠

**改动前**：地板默认 `contype=1, conaffinity=1`，与所有手部 geom 位掩码匹配，全部进入 AABB 检测。

### 2. 窄相位（Narrowphase）—— 性能瓶颈

`plane`（无限平面）的 AABB 是**无限大的**，所有手部 geom 的 AABB 都与之重叠，因此每一对都进入窄相位。

窄相位需要执行 **mesh vs plane** 的精确几何计算。即使最终没有实际接触，计算本身就非常昂贵。双手模型有几十个碰撞 mesh geom，**每个 timestep 都要做几十次 mesh-plane 精确距离计算**。

### 3. 改动后为什么快

```
contype/conaffinity 不匹配 → 直接跳过，连 AABB 都不算
```

位掩码检测只需一次整数位运算，几乎零成本。几十个碰撞对在最早期就被全部过滤。

### 类比

- **改动前**：每个包裹都拆开检查是否有违禁品（窄相位），最后发现全都没有
- **改动后**：看一眼面单标记就知道不用检查（位掩码），直接放行

## 其他可行的设置方式

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

## 总结

核心原则：**检测"没有碰撞"本身也在消耗算力**。在仿真场景中，应当通过碰撞掩码在最早期过滤掉不需要的碰撞对，尤其是涉及无限平面（plane）和大量碰撞 mesh 的场景。

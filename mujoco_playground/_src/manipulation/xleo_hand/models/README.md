# Models 工具脚本说明

本目录提供两个 mesh 处理脚本，用于为 MuJoCo/MJX 仿真准备碰撞几何体。

## 目录结构

```
models/
├── urdf/                        # URDF 模型文件
├── xmls/                        # MuJoCo XML (MJCF) 模型文件
├── ftl_meshes/                  # 原始 STL mesh 文件
│   └── convex/                  # 导出的凸包 mesh (由脚本生成)
├── export_convex_meshes.py      # 脚本1: 导出 MuJoCo 凸包
├── convex_decompose_urdf.py     # 脚本2: CoACD 凸分解
└── README.md
```

## 脚本 1: export_convex_meshes.py

**用途**: 从 MuJoCo 编译后的模型中导出凸包 mesh。MuJoCo 加载 STL 时会自动将碰撞 mesh 转为凸包，此脚本将这些凸包提取并保存为独立的 STL 文件。

**依赖**: `mujoco`, `trimesh`, `numpy`

**使用方法**:

```bash
# 直接运行，默认处理 xmls/ftl_xleo_dual_hand.xml
python export_convex_meshes.py
```

**输出**: 凸包 mesh 保存到 `ftl_meshes/convex/` 目录，文件名格式为 `{原名}_convex.stl`。

**示例输出**:
```
Model loaded: 11 meshes found
  LINK_HAND_BASE_L: 17989 verts, 36378 faces -> ftl_meshes/convex/LINK_HAND_BASE_L_convex.stl
  LINK_F0_L0: 3826 verts, 7624 faces -> ftl_meshes/convex/LINK_F0_L0_convex.stl
  ...
Done.
```

**适用场景**: 需要检查 MuJoCo 实际使用的碰撞形状，或将凸包 mesh 用于其他工具链。

## 脚本 2: convex_decompose_urdf.py

**用途**: 使用 [CoACD](https://github.com/SarahWeiii/CoACD) 对 URDF 中引用的碰撞 mesh 进行凸分解（Convex Decomposition）。与 MuJoCo 的单一凸包不同，CoACD 会将一个非凸 mesh 拆分为**多个**凸部件，更好地近似原始形状。

**依赖**: `coacd`, `trimesh`, `numpy`

**使用方法**:

```bash
# 默认参数，处理 urdf/ftl_xleo_dual_hand.urdf
python convex_decompose_urdf.py

# 自定义参数
python convex_decompose_urdf.py \
    --urdf urdf/ftl_xleo_dual_hand.urdf \
    --output-dir ftl_meshes \
    --threshold 0.03 \
    --max-convex-hull 16
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--urdf` | `urdf/ftl_xleo_dual_hand.urdf` | 输入 URDF 文件路径 |
| `--output-dir` | `ftl_meshes` | 凸分解 OBJ 文件输出目录 |
| `--threshold` | `0.05` | CoACD 凹度阈值，越小分解越精细 |
| `--max-convex-hull` | `32` | 每个 mesh 最大凸部件数 |

**输出**:
- 凸分解部件保存为 OBJ 文件: `ftl_meshes/{原名}_cvx_0.obj`, `{原名}_cvx_1.obj`, ...
- 新 URDF 文件: `urdf/ftl_xleo_dual_hand_convex.urdf`，碰撞体已替换为凸分解部件

## 两个脚本的区别

| | export_convex_meshes.py | convex_decompose_urdf.py |
|---|---|---|
| 输入 | MuJoCo XML (MJCF) | URDF |
| 方法 | MuJoCo 内置凸包 (单个凸包) | CoACD 凸分解 (多个凸部件) |
| 精度 | 粗略，单凸包可能丢失凹陷细节 | 精细，多凸部件更好保留原始形状 |
| 输出 | STL 文件 | OBJ 文件 + 新 URDF |
| 用途 | 快速检查/导出 MuJoCo 凸包 | 为仿真生成高质量碰撞几何体 |

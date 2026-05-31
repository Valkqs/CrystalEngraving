# 水晶内雕 · 立体点阵生成

从两张 2D 图片生成 3D 体素点阵：从立方体 **正视（-Z）** 方向看到第一张图，从 **侧视（-Y）** 方向看到第二张图。

## 算法原理

双视角投影体素雕刻：

1. 正视 / 侧视二值化后做 **交集拉伸**，再经 **双投影补全** 保证两向 OR 投影覆盖输入轮廓
2. **白边自动裁剪**：上传图片会先按有色区域自动提取最小外接矩形，再做等比缩放到画布，减少白边对重建的影响
3. **相接边等长对齐**：正视与侧视在裁剪后会按共享 X 边长度使用同一目标宽度缩放，先完成尺度对齐再进入体素生成
4. **深度双面补全**（默认开启）：侧视某列在 Z 向不连续时，不在整段深度填实，而是在该列 **最前、最后** 两个深度面各落点，减轻正视竖条空缺，同时尽量保留侧视中间的空隙
5. **边界四面包围补点**（默认开启）：当某个 X 列在另一视图没有覆盖时，回退到全局边界四面（`y_min/y_max/z_min/z_max`）补点，避免正视/侧视出现条纹空白
6. **侧视 Z 向填隙**（默认 2 px）：在侧视掩膜上闭合过小的 Z 向断档
7. **稀疏均匀化**：在保持两向投影的前提下减少点数

从 -Z 方向 OR 投影得到正视图；从 -Y 方向 OR 投影得到侧视图。

## 快速开始

**方式一（推荐，无需安装依赖）**

```bash
cd d:\Mountain\EngineerDesign
python serve.py
```

浏览器会自动打开 `http://localhost:8000/index.html`，体素算法在浏览器内运行。

**方式二（FastAPI 后端，适合 API 调用）**

```bash
pip install -r requirements.txt
python server.py
```

浏览器打开 [http://localhost:8000](http://localhost:8000)。若后端不可用，前端会自动回退到本地计算。

## 使用说明

1. 上传 **正视图** 与 **侧视图**（页面会预载圆形+矩形示例）
2. 调整分辨率、阈值；若图案为深色实体可勾选「反转」
3. 点击 **生成立体点阵**
4. 拖拽 3D 视图旋转，或使用视角按钮切换正视/侧视/俯视/等轴测
5. 右侧 **投影校验** 面板可对比重建投影与输入是否一致

## API

`POST /api/generate`

| 字段 | 类型 | 说明 |
|------|------|------|
| image_front | file | 正视图 |
| image_side | file | 侧视图 |
| size | int | 体素分辨率 16–1024，默认 192 |
| threshold | int | 手动二值化阈值（auto_threshold=false 时生效） |
| auto_threshold | bool | Otsu 自动阈值，默认 true |
| dilate | int | 形态学膨胀 0–5 px，默认 0 |
| align_x | bool | X 轴平移对齐，默认 true |
| clean_mask | bool | 去除孤立竖条噪点，精细模式默认 false |
| overlap_dilate | int | X 重叠扩展 0–3 px，默认 1 |
| depth_face_bridge | bool | 深度双面补全，默认 true |
| edge_wall_wrap | bool | 边界四面包围补点，默认 true |
| close_side_z_gaps | int | 侧视 Z 向填隙 0–8 px，默认 2 |
| density | float | 点阵密度 0.05–1.0，默认 0.75 |
| uniform_strength | float | 斜向均匀化 0–1，默认 0.25 |
| detail_mode | bool | 精细模式，默认 true |
| invert | bool | 是否反转 |

返回 JSON：`points`（体素坐标列表）、`count`、`projection_front`、`projection_side` 等。

## 技术栈

- 后端：Python、FastAPI、NumPy、Pillow
- 前端：Three.js、原生 HTML/CSS/JS

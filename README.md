# GelTilt — 凝胶电泳图像自动纠偏与泳道拼接工具

2026 华为开发者大赛 · 上海理工大学站 · 学生赛道（初赛）参赛作品，队伍 **B.W.T**。

GelTilt 读取凝胶成像仪拍摄的原始灰度图，用 GelGenie 分割条带、按泳道模型估计整幅图的倾斜角，
再对图像做刚性旋转使泳道恢复竖直；同时支持按条带区域自动裁切、按指定泳道顺序拼接，
以及一条命令行的批量处理与 JSON 报告。

## 仓库说明

本仓库为 GelTilt 的公开源码（与比赛提交包中的 `源码/` 目录一致，但去除了 `dist/` 预编译程序）。以下内容未纳入本仓库：

- 开发过程目录 `.codeartsdoer/` 与 `build/` 构建缓存；
- 比赛提交包内含**自包含单文件程序** `dist/geltilt-app.exe`（内置 GelGenie 模型与第三方许可证，约 130 MB）；本仓库不包含该二进制文件，可用 `python build_app_exe.py` 重新打包；
- 94 个基准样本的全部逐图对照图（`rot_output/*.png`、`discrepencies/*.png`），仅保留统计摘要
  `rot_output/summary.json`、`rot_output/discrepancies.json` 与 `discrepencies/index.txt`；
- GelGenie ONNX 模型权重默认不单独提供：比赛提交包内的 `dist/geltilt-app.exe` 已内置权重；源码运行首次使用时可自动下载，
  也可用环境变量 `GELTILT_GELGENIE_ONNX` 指定本地权重路径（见下文“模型准备”）。

`samples/` 收录 **10 张代表性真实凝胶样本**（全部基准为 94 个文件，其中一对文件字节相同，
去重后为 93 张独立凝胶）。完整基准的统计数字与拒绝记录见
[rot_output/summary.json](rot_output/summary.json)、[rot_output/discrepancies.json](rot_output/discrepancies.json)。

## 目录结构

```
GelTilt/（本仓库根目录）
├─ geltilt/                     # 主程序包（算法核心 + 命令行 + 桌面应用）
│  ├─ imageio.py                # 原生位深图像 I/O
│  ├─ rotation.py               # 刚性旋转
│  ├─ preprocess.py             # 预处理、极性判定、预览映射
│  ├─ projection.py             # 投影与锐度指标（保留工具模块）
│  ├─ bandmask.py               # GelGenie ONNX 推理（分块 + 羽化拼接）
│  ├─ banddetect.py             # 条带概率图 → 开运算 → 连通域 → 质心
│  ├─ ladders.py                # 泳道模型求角（唯一旋转估计算法）
│  ├─ smears.py                 # 涂抹/持久性估计（已退役，不参与默认路径）
│  ├─ viz.py                    # 纠偏前后 / 预处理前后对照图
│  ├─ cli.py / __main__.py      # 命令行入口
│  ├─ labeler.py                # 标注器与共享的 DPI/运行目录工具
│  ├─ projview.py               # 开发用投影查看器（可选）
│  └─ app/                      # 桌面应用（domain/ports/estimator/service/controller/widgets 分层）
├─ tests/                       # 274 个测试函数 / 301 个 pytest 用例
├─ samples/                     # 10 张代表性凝胶样本（1280×1024 灰度 TIFF）
├─ rot_output/                  # 基准统计摘要（summary.json、discrepancies.json）
├─ discrepencies/               # 偏差最大 25 个样本的索引
├─ app_runs.jsonl               # 应用层运行记录（55 条，§6.1 结论八来源）
├─ docs/tkinter_frontend/       # 桌面应用需求 / 设计 / 任务规范

├─ licenses/                    # 第三方许可证原文（含 Tcl/Tk、GelGenie）
├─ install.ps1                  # 一键安装脚本（venv + 依赖 + 自检 + 测试）
├─ run_app.py                   # 桌面应用入口
├─ build_app_exe.py             # PyInstaller 打包脚本（可选：build_exe.py / build_projview_exe.py）
├─ run_labeler.py / run_projview.py
├─ pyproject.toml / requirements.txt
├─ LICENSE / THIRD_PARTY_NOTICES.md
├─ SPEC-AUTOROTATE.md           # 自动纠偏算法规范
├─ SPEC-COMPOSITE.md            # 拼接式复合裁切规范
├─ BAND_DETECTION.md            # 条带检测流水线说明
├─ ground_truth.json            # 94 个基准样本的人工确认角度
└─ timeline.md                  # 开发时间线与决策记录
```

## 环境要求

- Windows 10 / 11 x64（算法核心跨平台；桌面应用使用 Tkinter）。
- Python ≥ 3.10（开发与打包验证版本：3.14.7）。
- 运行依赖：`numpy`、`pillow`、`opencv-python-headless`；测试另需 `pytest` 与 `scipy`（仅 `tests/test_smears.py`）。
- 打包依赖（可选）：`pyinstaller`。
- 显示：1024×768 及以上；支持高 DPI。

## 快速开始

一键安装（在仓库根目录打开 PowerShell）：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

手工安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pytest
```

### 模型准备（条带检测与真实样本测试需要）

代码按以下顺序查找 GelGenie 权重：

1. 环境变量 `GELTILT_GELGENIE_ONNX`（显式指定）；
2. 缓存目录 `~/.geltilt/gelgenie_universal_1024.onnx`；
3. 以上都不存在时，首次推理自动从 HuggingFace 下载到缓存目录（需联网一次，约 54.6 MB）。

离线部署：把权重放到 `~/.geltilt/gelgenie_universal_1024.onnx`（或设置 `GELTILT_GELGENIE_ONNX`）即可；
其 SHA-256 为 `56e335e59e8580adf9401b6a99744d11cfcae25462e4059a807e84981100ed29`。`dist/geltilt-app.exe` 已内置模型，无需下载。

模型就绪后，推理与界面完全离线。模型为 GelGenie 项目（Apache-2.0）原样再分发的权重，
不是本项目训练或修改的产物；详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 运行方式

### 桌面应用

```powershell
python run_app.py
# 无界面自检（构建窗口后立即退出，写出 app_selftest.json）
python run_app.py --selftest --run-dir .\selftest-run
# 指定启动目录
python run_app.py .\samples
```

应用内的“加载图像…”会打开单张图像，并扫描该图像所在文件夹；加载后用 ◀ / ▶ 切换样本。
自动旋转、手动微调、自动裁切、复合裁切、导出与会话日志的使用见作品说明文档第 3.1 节。

### 命令行

```powershell
# 帮助
geltilt --help

# 单张图像：纠偏 + JSON 报告
geltilt "samples\2026 0508 PPA6-1 PPA6-2 13772.Tif" --output-dir out\ --report out\report.json

# 多文件批量（CLI 不会递归扫描目录，请显式列出文件）
geltilt "samples\20260409 55130 MU-1.Tif" "samples\20260429 3.1 nhx.Tif" --output-dir out\

# 纠偏前后对照图
geltilt "samples\2026 0508 PPA6-1 PPA6-2 13772.Tif" --output-dir out\ --compare out\compare.png

# 只估计不写图
geltilt "samples\2026 0508 PPA6-1 PPA6-2 13772.Tif" --dry-run --report out\report.json
```

`--report` 在单输入时写一个 JSON 对象，多输入时写数组；字段包含 `input / ok / angle /
angle_reported / se / reason / warnings / kept / n_ladders / n_bands / fallback /
overdispersion / output / compare`，与桌面应用诊断面板一致；其中 `output` / `compare`
只在非 `--dry-run` 且实际写出文件时出现。

### 测试

```powershell
python -m pytest -q                 # 全量（预期 301 passed）
python -m pytest tests\app -q       # 应用层
python -m pytest -q -k lanes        # 泳道相关
```

说明：全量测试共 301 个用例。`tests/test_real_sample.py` 会用 `samples/` 中的真实样本调用完整估计器，
首次运行会下载模型（见“模型准备”）；离线环境请先按上文放置权重。

### 打包（可选）

```powershell
python -m pip install pyinstaller
python build_app_exe.py
.\dist\geltilt-app.exe --selftest
```

打包脚本会把 `~/.geltilt/`（或 `GELTILT_GELGENIE_ONNX` 指定）的模型内置进单文件程序；
随源码包提供的 `dist/geltilt-app.exe` 已内置模型与许可证，可完全离线运行。

## 样本与基准（10 张随包样本）

| # | 文件 | 人工确认角 | 基准估计角 | 说明 |
|---|------|-----------|-----------|------|
| 1 | `2026 0508 PPA6-1 PPA6-2 13772.Tif` | -2.8° | -2.942° | 测试硬编码样本（`tests/test_real_sample.py`） |
| 2 | `2026-519 胡椒rna2 f1r2 f2r1 rna1f2r2.Tif` | +1.4° | +3.161° | 中文文件名 / RNA 样本 |
| 3 | `20260409 55130 MU-1.Tif` | -2.9° | -2.981° | 单泳道、负倾角 |
| 4 | `20260416 65-1 42-1 13772.Tif` | +2.6° | +3.338° | 多泳道样本 |
| 5 | `20260420 caAAP6 55130-2 55130-2 13772-1 13772-2.Tif` | +1.1° | +3.026° | 多泳道样本 |
| 6 | `20260422 XP11-1 XP11-COLONY.Tif` | +2.9° | +3.602° | 多泳道、高不确定度示例（se 6.09°） |
| 7 | `20260429 3.1 nhx.Tif` | +3.0° | +4.399° | 单泳道、较大正倾角 |
| 8 | `20260430 990-CE 990-NE 3,1-CE 3.1-NE XHX-VE NHX-CE.Tif` | +0.9° | +0.332° | 多泳道一致性案例（8 条采纳泳道、71 个条带） |
| 9 | `20260514PEPPER RNA.Tif` | +1.4° | +1.924° | RNA 样本 |
| 10 | `20260518 3820 14130 6770.Tif` | -4.6° | -4.354° | 多泳道、较大负倾角 |

“基准估计角”取自完整基准运行 `rot_output/summary.json`（GelGenie + 泳道模型），
与测试使用的前端路径（`geltilt.core.estimate_rotation`）可能有小幅差异。

完整基准：94 个样本文件（去重后 93 张独立凝胶），其中 2 个被算法主动拒绝（拒绝是明确行为，
不是异常）。基准统计（与作品说明文档一致）：

- 成功给出角度：92 / 94（97.9%）；平均绝对误差 0.709°；中位绝对误差 0.6465°；
  最大绝对误差 1.926°；误差 ≤1° 占 75.0%，≤2° 占 100.0%。
- 报告标准误均值 / 最大：0.943° / 6.092°；短杠杆退化样本 7 / 92；
  泳道间过度离散 Q/dof 最大 4.75；每图条带数均值 34.3。
- 两个拒绝样本：
  `20260428 15810 990 酵母膜蛋白质粒.Tif`（no lane-like rung column found）、
  `20260811 62.Tif`（implausible tilt -8.1 deg exceeds +-8.0 deg; grouping suspect）。
- 已知重复：`20260423 131-CAAAP6-YFP.Tif` 与 `20260423 131-CAAAP6-YFP 1300-ATLUCKY.Tif`
  字节完全相同，基准统计按 94 个文件名计。

逐图复核方法：`rot_output/summary.json` 与 `ground_truth.json` 按文件名连接，逐条计算
|算法角度 − 人工确认角度|；`rot_output/discrepancies.json` 与 `discrepencies/index.txt`
给出偏差最大的 25 个样本。`app_runs.jsonl` 保留应用层 55 条运行记录（§6.1 结论八的来源），
随源码包一并提供便于复核。

## 文档

- [SPEC-AUTOROTATE.md](SPEC-AUTOROTATE.md)：自动纠偏的全部常量、处理顺序、门限与验收用例。
- [SPEC-COMPOSITE.md](SPEC-COMPOSITE.md)：拼接式复合裁切的泳道模型、规格语言与验收用例。
- [BAND_DETECTION.md](BAND_DETECTION.md)：条带检测流水线（模型约定、分块羽化、掩膜与质心）。
- [docs/tkinter_frontend/spec.md](docs/tkinter_frontend/spec.md)、`design.md`、`tasks.md`：桌面应用规范。
- [docs/improved-rotation-alg/](docs/improved-rotation-alg/)：最终旋转算法在独立实验中形成的规范与说明。
- [timeline.md](timeline.md)：带日期的开发决策与缺陷记录。

## 许可证与第三方组件

本项目以 MIT 许可证发布（见 [LICENSE](LICENSE)）。运行期使用 NumPy、Pillow、
OpenCV（headless）、Tcl/Tk 与 Python 运行时；打包使用 PyInstaller。各组件许可证原文见
[licenses/](licenses/)（含 `tcl-tk-license.terms`、`gelgenie-LICENSE.txt` 与 `gelgenie-NOTICE.txt`），
组件、版本、许可证与再分发义务见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

GelGenie 条带分割权重（Apache-2.0）由 University of Edinburgh 的研究者公开发布，
本项目未修改权重；引用信息：Aquilina et al., *Nature Communications* 16, 4087 (2025)。

## 验证记录

- Python 3.14.7；NumPy 2.5.3；Pillow 12.3.0；opencv-python-headless 5.0.0.93；
  PyInstaller 6.22.3；Tk 9.0。
- `python -m pytest -q`：301 passed（2026-10-19，源码包内实测）。
- `python run_app.py --selftest`：`ok=true`、`image_surfaces=1`、写出 `app_selftest.json`。
- 命令行单样本 `--dry-run --report` 可正常输出 JSON。
- `dist/geltilt-app.exe`（比赛提交包内）：自包含单文件构建，内置 54.6 MB 模型与许可证。

## 联系

队伍：B.W.T　|　作品：GelTilt（凝胶电泳图像自动纠偏工具）　|　版本：V1.2

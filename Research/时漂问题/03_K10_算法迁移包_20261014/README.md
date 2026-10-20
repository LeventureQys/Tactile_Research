# K10 算法迁移包（portable-1.1，2026-10-19）

在线双态蠕变观测器的**单文件可移植交付**。上位机（C/C++/C#/Python）与嵌入式（裸机/RTOS）
都从 `creep_observer_portable.h` 派生，不再各自维护算法副本。

- 权威口径：本目录的 `creep_observer_portable.h`
- 语义等价：`src/domain/drift_v6/creep_observer.cpp`（2026-10-19 已同步落地）
- 完整说明：`算法迁移文档.md`（逐帧规范 / 参数表 / 陷阱 / 验证）
- 缺陷背景：`../05_阶梯上飘诊断_20261019/诊断与修复报告.md`

---

## 1. 版本状态

| 项 | 值 |
|---|---|
| 版本 | **portable-1.2** |
| 相对 1.0 的增量 | ① `floor_rise_idle_only`（y_floor 抬升门）② `guard_floor_recover_s`（下探回抬锚）——**默认都是 0** |
| 默认行为 | 与 portable-1.0 **逐帧一致**（287374 帧 × 57 通道，`guard=0/1` 与 `recover=0/30` 四臂实测差 0.0） |
| 产品侧状态 | `Params::floor_rise_idle_only = 1`（修复态默认开）、`guard_floor_recover_s = 0`（待定档），已与产品源码逐帧对拍 0.0 |
| 固件下发 | **未下发** |

`floor_rise_idle_only = 1` 修的是"长恒载把载荷吸收进基线 → idle 判据被噪声击穿 → 补偿永久归零
→ 显示阶梯式上飘"。同一份 48 min 实录：漂移 **+3497 → −74 ADC**，台阶数 **30 → 3**，
`applied` 最小值 **599 → 3932**。**K10 的防护层（`guard_enable=1`）拦不住这条路径**，见报告 §3。

`guard_floor_recover_s = 30` 修的是抬升门留下的另一半问题——`y_floor` 被真实下探拉低后不再回抬
（实测末值 = 最小值：−758.9 / −654.2）。开启后回抬到 **−249**，主指标不变。**但收益尚未被证实**
（缺"长恒载 → 卸载 → 空载"录制），故默认关闭，详见迁移文档 §9.6。

---

## 2. 文件

| 文件 | 说明 |
|---|---|
| `creep_observer_portable.h` | ★ 唯一算法文件（头 + 实现，stb 风格单文件） |
| `算法迁移文档.md` | ★ 零上下文迁移文档（逐帧规范 / 参数表 / 已知陷阱 / 验证方法） |
| `dv6_portable_abi.c` | C ABI 包装层（仅 FFI 需要，不含算法逻辑） |
| `build_dll.bat` | MSVC 构建脚本（**必须带 `/utf-8`**，见 §4） |
| `dv6_portable.dll` | 已编译产物（MSVC 2022，`/utf-8 /O2`） |
| `verify_portable.py` | 关闭档对拍工具箱参考实现 + 生成黄金向量（含 gate=1 契约） |
| `guard_parity.py` | 防护层开启档对拍 Python 参考实现 |
| `verify_result.json` / `guard_parity_result.json` | 上述两项的实测结果 |
| `golden_vectors/` | 输入 / 关闭档期望输出 / gate=1 期望输出 / 参数 + 契约说明 |
| `creep_observer_portable.k9_20261014.h.bak` | portable-1.0 头文件留档（回滚用） |
| `dv6_portable_k9_20261014.dll.bak` | portable-1.0 DLL 留档（回滚用） |

---

## 3. 两个可选开关

| 参数 | 默认 | 作用 | 何时开 |
|---|---:|---|---|
| `guard_enable` | 0 | K10 瞬态防护：`y_floor` 下行确认 + 快速泄放确认 | 现场存在短促下探污染基线时（见 §9 防护层） |
| `floor_rise_idle_only` | 0（本包）/ 1（产品） | `y_floor` 抬升门：只在判为空载时抬升 | **长恒载场景必须开**；不开会在 20 min 后崩塌 |
| `guard_floor_recover_s` | 0 | 下探回抬锚：回到下探起点之上时把 `y_floor` 抬回锚 | 需要"卸载→空载"段验证收益后再定档，建议从 30 起 |

三者**正交**，可任意组合。`gate=1 + guard=1` 与 `gate=1` 在 48 min 实录上差 ≤4 ADC；
`recover=30` 不改变漂移/p2p/`applied`，只把 `y_floor` 从 −666 回抬到 −249。

---

## 4. 复现与自检

```powershell
cd "D:\workshop\Github\Tactile_Research\Research\时漂问题\03_K10_算法迁移包_20261014"

# 1) 重新编译（改过 .h 才需要）
build_dll.bat

# 2) 关闭档对拍 + 重新生成黄金向量
python verify_portable.py
#    期望：params_bytes {dll:256, ctypes:256}，max_abs_diff = 0.0，equivalent = true

# 3) 防护层开启档对拍
python guard_parity.py
#    期望：max_abs_diff = 0.0，equivalent = true
```

依赖：工具箱参考实现 `D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具`，
录制数据 `Research\时漂问题\data\archived\...`（脚本内已写绝对路径）。

**两条静默失败点**（不报错，但行为全错）：

1. **`dv6_params` 布局失配**：加字段后 `sizeof` 248 → 256。FFI 侧仍按旧布局传会让 C 读到缓冲区外
   4 字节。两个验证脚本已内置 `dv6_portable_params_bytes()` vs `ctypes.sizeof` 校验，移植时照做。
2. **编译缺 `/utf-8`**：源文件是 UTF-8 无 BOM，MSVC 默认按 936/GBK 解析会误读中文注释
   （C4819）。上位机工程 `CMakeLists.txt` 已全局 `/utf-8`，本包 `build_dll.bat` 已对齐。

---

## 5. 回滚

单分支加锁，**改参数即可回滚，无需回退代码**：把 `floor_rise_idle_only` 置 0。
需要整体回到 portable-1.0 时，用上表两个 `.bak` 覆盖同名文件。

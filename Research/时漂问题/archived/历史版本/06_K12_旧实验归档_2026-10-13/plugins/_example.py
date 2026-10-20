# -*- coding: utf-8 -*-
"""示例算法插件（教学模板）——下划线开头的文件默认不会被扫描加载。

把它复制成 `plugins/<你的插件id>.py` 即可接入工具。插件是一个**单文件 Python 模块**，
只需提供三件套：

  1. PLUGIN_META（dict）——模块级元数据：
       id          必填，全局唯一，只允许小写字母/数字/下划线且以字母开头：^[a-z][a-z0-9_]*$
       name        必填，显示名（Dialog 下拉框里显示）
       version     必填，插件自身版本（如 "1.0.0"）
       protocol    必填，协议版本，必须等于 dptool.algorithms.PLUGIN_PROTOCOL_VERSION
       input_mode  必填，"dataset"（吃整张会话表：时间轴 + 全通道矩阵 + meta）
                            或 "curves"（吃用户在左侧选中的曲线列表）
       description 选填，Dialog 里的说明文字
       time_modes  选填，支持的时间轴模式子集，默认全部 ["frame_fps", "elapsed", "timestamp"]
       params      示例插件声明了 3 个可见参数（窗口/缩放/平移）+ 1 个 hidden 参数（warmup），
                   default 必须通过自身的类型/范围校验

  2. param_specs() -> tuple[ParamSpec, ...]——参数 schema，Dialog 按它自动生成表单。
       每条 ParamSpec 的字段：
         name/label/type/default 必填；type ∈ {"float", "int", "bool", "enum"}
         float/int 必须给 min 与 max；enum 必须给 choices；bool 只看 default
         step/decimals 控制旋钮步进与小数位（float 不给 decimals 时按 step 推）
         unit/help/group 只影响显示（help 显示为控件 tooltip，group 是 QGroupBox 分组名）
         hidden=True 只表示该参数不渲染进 UI 表单（给内部/高级参数用），它仍是完全合法的
         参数：default_params 会带上它、validate_params 照常校验它、可传给 run、可往返配置 JSON

  3. class Algorithm——算法本体，必须实现：
         def run(self, ctx: AlgorithmContext, params: dict, progress=None, cancel=None)
             -> AlgorithmResult

协议红线（宿主不代你做检查，违反会被记为运行失败）：
  - 算法必须**无状态**：所有状态在 run 内部创建，同一个实例可能被复用多次 run；
  - 禁止 import 任何 Qt/PySide6 模块，也禁止触碰界面对象：run 跑在 worker 线程上；
  - 不得修改传入的 params dict；
  - progress(frac, msg) 主动上报 0.0~1.0 进度；cancel() 返回 True 时必须尽快抛
    dptool.algorithms.Cancelled（或返回空结果，宿主按取消丢弃）；
  - 每条 AlgorithmOutput.y 的长度必须等于 len(ctx.time_axis)，outputs 至少 1 条；
  - 输出允许含 NaN（绘制层已兼容）。

本示例（curves 模式）实现 y = offset + scale * moving_average(x, window)。
"""
import numpy as np

from dptool.algorithms import AlgorithmOutput, AlgorithmResult, Cancelled, ParamSpec

PLUGIN_META = {
    "id": "example_moving_average",
    "name": "示例·滑动平均",
    "version": "1.0.0",
    "protocol": 1,
    "description": "教学模板：对每条输入曲线做移动平均并按系数缩放（curves 模式）",
    "input_mode": "curves",
    "time_modes": ["elapsed", "timestamp", "frame_fps"],
}


def param_specs():
    return (
        ParamSpec(name="window", label="窗口点数", type="int", default=15, min=1, max=501, step=2,
                  unit="点", help="移动平均窗口，越大越平滑", group="平滑"),
        ParamSpec(name="scale", label="缩放系数", type="float", default=1.0, min=-100.0, max=100.0,
                  step=0.01, decimals=3, unit="×", help="输出 = 平移 + 缩放 × 平滑值", group="标定"),
        ParamSpec(name="offset", label="平移量", type="float", default=0.0, min=-10000.0, max=10000.0,
                  step=0.1, decimals=3, unit="", help="常数平移，便于对齐纵轴", group="标定"),
        # hidden=True：该参数仅内部/高级使用，不进 UI 表单；但仍是合法参数（进 default_params、
        # 照常参与 validate_params 校验、可传入 run，也能随配置 JSON 导出导入）
        ParamSpec(name="warmup", label="预热帧数", type="int", default=0, min=0, max=100000,
                  step=1, unit="帧", help="保留参数，示例不参与计算", group="高级", hidden=True),
    )


class Algorithm:
    def run(self, ctx, params, progress=None, cancel=None):
        window = int(params.get("window", 15))
        scale = float(params.get("scale", 1.0))
        offset = float(params.get("offset", 0.0))
        # 隐藏参数与可见参数一样从这里取值（本示例保留但不参与计算）
        int(params.get("warmup", 0))
        outputs = []
        total = max(1, len(ctx.curves))
        for index, curve in enumerate(ctx.curves):
            if cancel is not None and cancel():
                raise Cancelled()
            if progress is not None:
                progress(index / total, "处理 %s" % curve.name)
            source = np.asarray(curve.y, dtype=np.float64)
            smoothed = _moving_average(source, window)
            outputs.append(AlgorithmOutput(key=curve.curve_id, label="示例·%s" % curve.name,
                                           y=offset + scale * smoothed, unit=""))
        if progress is not None:
            progress(1.0, "完成")
        note = "curves 模式；窗口 %d 点，缩放 %g，平移 %g" % (window, scale, offset)
        return AlgorithmResult(outputs=outputs, note=note, params_used=dict(params))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """前缀和实现，O(n) 时间；NaN 会扩散到覆盖它的窗口内（示例刻意保持简单）。"""
    span = max(1, int(window))
    if values.size == 0:
        return values
    if span <= 1:
        return values
    pad = span // 2
    padded = np.concatenate((np.full(pad, values[0], dtype=np.float64), values,
                             np.full(span - 1 - pad, values[-1], dtype=np.float64)))
    cumulative = np.concatenate(([0.0], np.cumsum(padded)))
    return (cumulative[span:] - cumulative[:-span]) / float(span)


__all__ = ["PLUGIN_META", "param_specs", "Algorithm"]

"""生成对比图（PIL 绘制，无需 matplotlib）。"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import core

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

FONTS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhl.ttc", r"C:\Windows\Fonts\simhei.ttf"]


def font(size):
    for p in FONTS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def load_cells(name):
    path = ROOT / name
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            skip += 1
            if line.startswith("##Data"):
                break
        skip += 1
    df = pd.read_csv(path, skiprows=skip, header=None, engine="c")
    return df.iloc[:, 0].to_numpy(dtype=np.float64), df.iloc[:, 3:].to_numpy(dtype=np.float64)


def resample(t, x, ds):
    b = np.floor((t - t[0]) / ds).astype(np.int64)
    n = b[-1] + 1
    s = np.bincount(b, weights=x, minlength=n)
    c = np.bincount(b, minlength=n)
    return np.arange(n) * ds, s / np.maximum(c, 1)


class Panel:
    def __init__(self, img, box, xr, yr, title, xlabel, ylabel):
        self.d = ImageDraw.Draw(img)
        self.x0, self.y0, self.x1, self.y1 = box
        self.xr, self.yr = xr, yr
        self.d.rectangle(box, outline=(210, 214, 220))
        f = font(19)
        self.d.text((self.x0, self.y0 - 30), title, font=f, fill=(24, 28, 36))
        f2 = font(14)
        # y 轴刻度
        for k in range(6):
            v = yr[0] + (yr[1] - yr[0]) * k / 5
            yy = self.y1 - (self.y1 - self.y0) * k / 5
            self.d.line([(self.x0 - 5, yy), (self.x0, yy)], fill=(140, 146, 156))
            self.d.text((self.x0 - 62, yy - 9), f"{v:,.0f}", font=f2, fill=(90, 96, 106))
        for k in range(7):
            v = xr[0] + (xr[1] - xr[0]) * k / 6
            xx = self.x0 + (self.x1 - self.x0) * k / 6
            self.d.line([(xx, self.y1), (xx, self.y1 + 5)], fill=(140, 146, 156))
            self.d.text((xx - 18, self.y1 + 9), f"{v:,.0f}", font=f2, fill=(90, 96, 106))
        self.d.text(((self.x0 + self.x1) / 2 - 60, self.y1 + 32), xlabel, font=f2, fill=(60, 66, 76))
        self.d.text((self.x0 - 62, self.y0 - 54), ylabel, font=f2, fill=(60, 66, 76))

    def xy(self, x, y):
        px = self.x0 + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.x1 - self.x0)
        py = self.y1 - (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * (self.y1 - self.y0)
        return px, py

    def line(self, xs, ys, color, width=2):
        pts = [self.xy(x, y) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            self.d.line(pts, fill=color, width=width, joint="curve")

    def vline(self, x, color, dash=8):
        px, _ = self.xy(x, self.yr[0])
        y = self.y0
        while y < self.y1:
            self.d.line([(px, y), (px, min(y + dash, self.y1))], fill=color, width=2)
            y += dash * 2

    def legend(self, items, pos=None):
        f = font(15)
        x = pos[0] if pos else self.x0 + 16
        y = pos[1] if pos else self.y0 + 12
        for label, color in items:
            self.d.line([(x, y + 8), (x + 26, y + 8)], fill=color, width=3)
            self.d.text((x + 32, y), label, font=f, fill=(40, 46, 56))
            y += 24


def main():
    ts, pre = load_cells("device_001_pre_seg0.csv")
    _, algoadc = load_cells("device_001_algoadc_seg0.csv")
    t = ts - ts[0]
    v9 = core.run(pre, ts, {})
    vg = core.run(pre, ts, {"floor_rise_gate": 1})
    v10 = core.run(pre, ts, {"guard_enable": 1})
    tsec, raw_s = resample(t, pre.sum(1), 1.0)
    _, a9 = resample(t, v9.sum(1), 1.0)
    _, a10 = resample(t, v10.sum(1), 1.0)
    _, ag = resample(t, vg.sum(1), 1.0)
    np.savez_compressed(OUT / "compare.npz", tsec=tsec, raw=raw_s, k9=a9, k10=a10, gate=ag,
                        ap9=raw_s - a9, ap10=raw_s - a10, apg=raw_s - ag)

    W, H = 1500, 1080
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((60, 22), "拇指指腹 2026-10-19 16:33（48 min 恒载）：K9/K10 阶梯上飘 与 修复方案",
           font=font(25), fill=(16, 20, 28))
    d.text((60, 58), "输入（原始 ADC 总和）只漂 +314 ADC；现役 K9 输出漂 +3495 ADC——漂移是算法自伤",
           font=font(16), fill=(90, 96, 106))

    top = Panel(img, (150, 130, W - 60, 520), (0, 2865), (27000, 33000),
                "① 补偿后总力（ADC 总和）", "时间 (s)", "ADC")
    top.line(tsec, raw_s, (150, 156, 166), 2)
    top.line(tsec, a9, (214, 69, 65), 2)
    top.line(tsec, a10, (240, 150, 60), 2)
    top.line(tsec, ag, (35, 130, 200), 2)
    top.vline(1600, (120, 120, 120))
    top.legend([("原始输入 pre", (150, 156, 166)), ("K9 现役（本次录制实跑）", (214, 69, 65)),
                ("K10 防护 guard=1", (240, 150, 60)), ("修复：y_floor 仅空载抬升", (35, 130, 200))])
    d.text((top.xy(1600, 27000)[0] + 6, 500), "t=1600s 起逐通道崩塌", font=font(14), fill=(80, 86, 96))

    bot = Panel(img, (150, 640, W - 60, 1000), (0, 2865), (-300, 4600),
                "② 算法实际扣掉的补偿量 applied = pre − out", "时间 (s)", "ADC")
    bot.line(tsec, raw_s - a9, (214, 69, 65), 2)
    bot.line(tsec, raw_s - a10, (240, 150, 60), 2)
    bot.line(tsec, raw_s - ag, (35, 130, 200), 2)
    bot.vline(1600, (120, 120, 120))
    bot.legend([("K9 现役", (214, 69, 65)), ("K10 防护", (240, 150, 60)), ("修复", (35, 130, 200))],
               pos=(W - 400, 660))
    img.save(FIG / "图1_全程对比.png")

    # 图2：18~34min 放大
    m = (t >= 1060) & (t <= 2060)
    W2, H2 = 1500, 780
    img2 = Image.new("RGB", (W2, H2), (255, 255, 255))
    d2 = ImageDraw.Draw(img2)
    d2.text((60, 20), "18~34 min 放大：阶梯从哪来（t=1599.977s 首帧击穿，之后逐通道接续）",
            font=font(23), fill=(16, 20, 28))
    p = Panel(img2, (150, 120, W2 - 60, 640), (1060, 2060), (27500, 33000),
              "补偿后总力（ADC 总和）", "时间 (s)", "ADC")
    tm = t[m]
    p.line(tm, pre.sum(1)[m], (150, 156, 166), 1)
    p.line(tm, v9.sum(1)[m], (214, 69, 65), 2)
    p.line(tm, vg.sum(1)[m], (35, 130, 200), 2)
    p.vline(1600, (120, 120, 120))
    p.legend([("原始输入", (150, 156, 166)), ("K9 现役", (214, 69, 65)), ("修复", (35, 130, 200))])
    img2.save(FIG / "图2_18-34min放大.png")

    # 图3：逐通道 applied 塌陷时刻（阶梯本质）
    v, L, Lb, Ls = core.run(pre, ts, {}, log=True)
    ap = L["applied"]
    base = ap[(t > 900) & (t < 1000)].mean(0)
    order = np.argsort(base)[::-1][:40]
    W3, H3 = 1500, 900
    img3 = Image.new("RGB", (W3, H3), (255, 255, 255))
    d3 = ImageDraw.Draw(img3)
    d3.text((60, 20), "40 个最强通道的 applied：每条线是一个通道，掉下去的时刻各不相同 → 阶梯",
            font=font(23), fill=(16, 20, 28))
    p3 = Panel(img3, (150, 120, W3 - 60, 800), (0, 2865), (-10, 320),
              "逐通道 applied（ADC）", "时间 (s)", "ADC")
    step = int(1.0 / np.median(np.diff(t)))
    for k in order:
        p3.line(t[::step], ap[::step, k], (60 + (k * 37) % 150, 90 + (k * 53) % 120,
                                           90 + (k * 71) % 110), 1)
    p3.vline(1600, (60, 60, 60))
    img3.save(FIG / "图3_逐通道阶梯.png")
    print("图已写:", *[str(p) for p in sorted(FIG.glob("*.png"))], sep="\n  ")


if __name__ == "__main__":
    main()


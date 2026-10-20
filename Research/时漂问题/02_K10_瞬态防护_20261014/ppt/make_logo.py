# -*- coding: utf-8 -*-
"""把 Modutech 的 .ico 转为 PPT 可直接使用的 PNG（透明背景，2x 备用尺寸）。"""
from pathlib import Path

from PIL import Image

SRC = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\assets\logo.ico")
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

ico = Image.open(SRC)
ico.size = (256, 256)
best = ico.convert("RGBA")

# 统一放到 1024 画布上等比放大，保证 PPT 缩放时边缘干净
target = 1024
canvas = Image.new("RGBA", (target, target), (0, 0, 0, 0))
scaled = best.resize((target, target), Image.LANCZOS)
canvas.alpha_composite(scaled)
canvas.save(OUT / "modutech_logo.png")

small = best.resize((256, 256), Image.LANCZOS)
small.save(OUT / "modutech_logo_256.png")

print("source frames:", sorted(ico.info.get("sizes", [])))
print("written:", (OUT / "modutech_logo.png").stat().st_size,
      (OUT / "modutech_logo_256.png").stat().st_size)

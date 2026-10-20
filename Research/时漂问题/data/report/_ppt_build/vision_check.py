# -*- coding: utf-8 -*-
"""用 DeepSeek 视觉模型审核配图 / PPT 渲染页。"""
from __future__ import annotations

import base64
import json
import mimetypes
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRED = Path(r"C:\Users\lenovo\.dsh\.credentials.yaml")
ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash-vision-exp"

FIGURE_PROMPT = """你是资深技术汇报图审阅员。这张图会放进给甲方（客户）的 PPT 里。请如实报告你看到的内容。

逐条回答，不得合并：
1. 图中有哪几条曲线/柱形，颜色分别是什么，图例文字写的是什么；
2. 曲线/柱子是否被画布边缘裁切，是否有任何文字被裁切或压到画外；
3. 图例是否遮挡曲线或柱子；文字是否互相重叠；
4. 中文是否正常显示（不是方块、问号、乱码）；
5. 坐标轴标题、刻度单位是否完整可读；
6. 看图上的数值标注，写下你能读到的所有数字；
7. 是否有明显空白失衡或元素挤在一起；
8. 这张图能否直接放进客户 PPT（可 / 不可），不可则指出最严重问题。

输出格式：
【内容】每项一行，写"通过"或"问题：<位置> <证据>"。
【数字】你在图上读到的全部数值。
【结论】可交付 / 需修改。"""


def read_key() -> str:
    text = CRED.read_text(encoding="utf-8")
    m = re.search(r"DEEPSEEK_API_KEY:\s*(\S+)", text)
    if not m:
        raise SystemExit("DEEPSEEK_API_KEY not found")
    return m.group(1).strip().strip('"').strip("'")


def ask(key: str, image: Path, prompt: str) -> dict:
    mime = mimetypes.guess_type(image.name)[0] or "image/png"
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode()}},
            ],
        }],
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return {"image": image.name, "usage": body.get("usage"),
            "content": body["choices"][0]["message"]["content"]}


def main() -> int:
    args = sys.argv[1:]
    prompt = FIGURE_PROMPT
    out_name = "vision_review.json"
    if args and args[0] == "--slides":
        prompt = (FIGURE_PROMPT.replace("这张图", "这张幻灯片渲染图")
                  .replace("曲线/柱子", "曲线/图文元素"))
        args = args[1:]
        out_name = "vision_slides.json"
    if args and args[0] == "--dir":
        images = sorted((HERE / args[1]).glob("*.png"))
    else:
        images = [Path(a) if Path(a).is_absolute() else HERE / a for a in args]

    key = read_key()
    results = []
    for image in images:
        try:
            results.append(ask(key, image, prompt))
            print(f"[ok] {image.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"image": image.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[fail] {image.name}: {exc}", flush=True)
    (HERE / out_name).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for item in results:
        print("\n" + "=" * 78 + f"\n### {item['image']}")
        print(item.get("content") or item.get("error"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

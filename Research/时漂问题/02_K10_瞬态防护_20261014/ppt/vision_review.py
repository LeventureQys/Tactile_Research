# -*- coding: utf-8 -*-
"""用 DeepSeek 视觉模型审核 PPT 渲染图。

用法：
  python vision_review.py <图片路径> [更多图片...]
  python vision_review.py --dir preview-v1            # 审核目录内全部页面

把每张图单独发一次请求（一次一图，符合逐张读图的要求），
结果以 JSON 落盘到 vision_review.json。
"""
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

PROMPT = """你是资深 PPT 版面审阅员。请打开这张幻灯片渲染图，逐项检查并如实报告。

检查项（逐条回答，不得合并、不得省略）：
1. 文字是否溢出所在色块、卡片、表格单元格或文字框；
2. 文字是否与图片、图例、其他文字重叠；
3. 文字是否被页面边缘裁掉或贴边到不可接受的程度；
4. 图片图表内部是否有文字穿框、图例遮挡曲线、坐标轴或数据被裁切；
5. 中文是否有乱码、缺字、显示成方块；
6. 是否存在明显空白失衡（某一大块区域空着 / 内容挤在一起）；
7. 表格文字是否被单元格换行切断或行高不足导致压线；
8. 整体上这一页能否直接拿去给客户看（可 / 不可），不可则说明最严重的问题。

输出格式：
【逐项】每项一行，写"通过"或"问题：<位置> <你看到的证据>"。位置必须具体（例如"左下卡片第二行"）。
【必须修改】列出必须修改的问题，按严重程度排序；没有就写"无"。
【结论】可交付 / 需修改。"""


def read_key() -> str:
    text = CRED.read_text(encoding="utf-8")
    m = re.search(r"DEEPSEEK_API_KEY:\s*(\S+)", text)
    if not m:
        raise SystemExit("DEEPSEEK_API_KEY not found in credentials")
    return m.group(1).strip().strip('"').strip("'")


def ask(key: str, image: Path, extra: str = "") -> dict:
    mime = mimetypes.guess_type(image.name)[0] or "image/png"
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT + (("\n\n补充说明：" + extra) if extra else "")},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode()}},
            ],
        }],
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return {
        "image": image.name,
        "model": body.get("model"),
        "usage": body.get("usage"),
        "content": body["choices"][0]["message"]["content"],
    }


def main() -> int:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "--dir":
        images = sorted((HERE / args[1]).glob("*.png"))
    else:
        images = [Path(a) if Path(a).is_absolute() else HERE / a for a in args]

    key = read_key()
    results = []
    for image in images:
        try:
            results.append(ask(key, image))
            print(f"[ok] {image.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"image": image.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[fail] {image.name}: {exc}", flush=True)

    out = HERE / "vision_review.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for item in results:
        print("\n" + "=" * 78)
        print("### " + item["image"])
        print(item.get("content") or item.get("error"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

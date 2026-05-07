#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Hugo frontmatter-style tags from a short plot / outline (Japanese).

Usage:
  python scripts/generate_tags.py --plot "MGS2の思い出とPS2時代の話"
  python scripts/generate_tags.py < plot.txt
  Get-Content plot.txt -Raw | python scripts/generate_tags.py

With LLM (recommended for Japanese):
  set OPENAI_API_KEY=...
  python scripts/generate_tags.py --plot "..." --model gpt-4o-mini

PowerShell で引数が壊れる場合は --plot=... 形式か、プロットをファイルにして
  python scripts/generate_tags.py --plot-file myplot.txt

Outputs a line you can paste into YAML frontmatter:
  tags: ["日記","ゲーム"]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any


def _read_plot(args: argparse.Namespace) -> str:
    if args.plot_file:
        with open(args.plot_file, encoding="utf-8") as f:
            return f.read().strip()
    if args.plot:
        return args.plot.strip()
    if sys.stdin.isatty():
        print(
            "プロットを --plot または --plot-file で渡すか、標準入力に貼り付けてください。",
            file=sys.stderr,
        )
        sys.exit(2)
    return sys.stdin.read().strip()


def _yaml_escape(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def format_tags_line(tags: list[str]) -> str:
    inner = ",".join(_yaml_escape(t) for t in tags)
    return f"tags: [{inner}]"


def heuristic_tags(text: str, max_tags: int) -> list[str]:
    """No-API fallback: bullets, hashtags, 「」, and explicit タグ lines."""
    seen: set[str] = set()
    out: list[str] = []

    def push(s: str) -> None:
        s = s.strip()
        if not s or len(s) > 32:
            return
        if s in seen:
            return
        seen.add(s)
        out.append(s)

    for m in re.finditer(r"#(\S{1,24})", text):
        push(m.group(1))

    for m in re.finditer(r"「([^」]{1,24})」", text):
        push(m.group(1))

    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^タグ\s*[:：]", line):
            rest = re.sub(r"^タグ\s*[:：]\s*", "", line)
            for part in re.split(r"[、,，]\s*", rest):
                push(part)
            continue
        m = re.match(r"^[・\-\*]\s*(.+)$", line)
        if m and len(m.group(1)) <= 24:
            push(m.group(1))

    return out[:max_tags]


def llm_tags(
    plot: str,
    *,
    api_key: str,
    model: str,
    min_tags: int,
    max_tags: int,
) -> list[str]:
    system = (
        "あなたはブログの編集補助です。与えられたプロット（箇条書きやメモ可）から、"
        "記事に付けるタグだけを選びます。ルール:\n"
        "- 日本語の短い名詞・固有名詞・ジャンル（例: 日記, ゲーム, 小説）を使う\n"
        "- 冗長な説明文にしない（1タグ20文字以内目安）\n"
        "- 意味が重複するタグはまとめる\n"
        f"- 個数は{min_tags}個以上{max_tags}個以下\n"
        '返答はJSONオブジェクトのみ: {"tags":["..."]}'
    )
    user = f"プロット:\n{plot}"
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.4,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {e.code}: {err}") from e

    try:
        content = payload["choices"][0]["message"]["content"]
        data = json.loads(content)
        tags = data.get("tags") or []
        if not isinstance(tags, list):
            raise ValueError("tags is not a list")
        cleaned = [str(t).strip() for t in tags if str(t).strip()]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise RuntimeError(f"Unexpected API response: {payload!r}") from e

    if not cleaned:
        raise RuntimeError(f"Empty tags from model: {payload!r}")
    return cleaned[:max_tags]


def main() -> None:
    p = argparse.ArgumentParser(description="プロットからHugo用tags行を生成")
    p.add_argument("--plot", help="プロット本文（短いメモで可）")
    p.add_argument("--plot-file", help="プロットが入ったUTF-8テキストファイル")
    p.add_argument("--min-tags", type=int, default=2)
    p.add_argument("--max-tags", type=int, default=8)
    p.add_argument(
        "--model",
        default=os.environ.get("OPENAI_TAG_MODEL", "gpt-4o-mini"),
        help="Chat Completions モデル名（既定: gpt-4o-mini）",
    )
    p.add_argument(
        "--heuristic-only",
        action="store_true",
        help="LLMを使わずルールベースのみ（精度は低め）",
    )
    p.add_argument(
        "--json-out",
        action="store_true",
        help="tags配列を含むJSONオブジェクトのみを出力",
    )
    args = p.parse_args()

    plot = _read_plot(args)
    if not plot:
        print("プロットが空です。", file=sys.stderr)
        sys.exit(2)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    tags: list[str]

    if not args.heuristic_only and api_key:
        try:
            tags = llm_tags(
                plot,
                api_key=api_key,
                model=args.model,
                min_tags=args.min_tags,
                max_tags=args.max_tags,
            )
        except Exception as e:
            print(f"LLMタグ生成に失敗しました: {e}", file=sys.stderr)
            print("ヒューリスティックにフォールバックします。", file=sys.stderr)
            tags = heuristic_tags(plot, args.max_tags)
    else:
        if not args.heuristic_only and not api_key:
            print(
                "OPENAI_API_KEY が未設定のため、ヒューリスティック抽出のみ行います。"
                "精度を上げるには API キーを設定してください。",
                file=sys.stderr,
            )
        tags = heuristic_tags(plot, args.max_tags)

    if not tags:
        print(
            "タグが1件も抽出できませんでした。"
            "プロットに「タグ: A, B」や #キーワード を書くか、OPENAI_API_KEY を設定してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.json_out:
        print(json.dumps({"tags": tags}, ensure_ascii=False))
    else:
        print(format_tags_line(tags))


if __name__ == "__main__":
    main()

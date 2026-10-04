"""Step 1：把《京东集团员工手册.pdf》转为可编辑文本 + 结构化逐页记录。

产物：
  data/handbook_editable.md  整册可编辑 Markdown（逐页正文 + 表格 + 页标记）
  data/pages.jsonl           逐页结构化 JSONL（{page,text,tables,has_image,is_cover,is_toc,char_len}）
  data/stats.json            统计报告（页数 / 字符量 / 空页 / 表格页 / 剔除噪声行数）

用法（务必用 InternVL 环境）：
  & "C:\\Users\\wuhao\\.conda\\envs\\InternVL\\python.exe" -m ragcore.extract
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime

import pdfplumber

from .config import (
    DATA_DIR,
    EDITABLE_MD,
    NOISE_MAX_EDGE,
    NOISE_MIN_PAGE_FREQ,
    PAGES_JSONL,
    PDF_PATH,
    STATS_JSON,
)


def ensure_utf8() -> None:
    """规避 Windows 控制台 GBK 编码导致的乱码/崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def _norm(line: str) -> str:
    return line.strip()


def _is_toc_page(text: str) -> bool:
    """目录页判定：页面开头出现『目录』（容忍空格）。"""
    head = text[:200].replace(" ", "").replace("\u3000", "")
    return "目录" in head


def _sanitize_cell(value) -> str:
    """清洗表格单元格：换行折叠为空格、转义竖线。"""
    if value is None:
        return ""
    s = str(value).replace("\r", " ").replace("\n", " ").strip()
    return s.replace("|", "\\|")


def table_to_markdown(table: list) -> str:
    """pdfplumber 表格 -> GFM 表格（含表头分隔行）。"""
    if not table:
        return ""
    rows = [[_sanitize_cell(c) for c in row] for row in table]
    rows = [r for r in rows if any(cell for cell in r)]
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |"]
    out.append("| " + " | ".join(["---"] * ncols) + " |")
    for row in rows[1:]:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _inside_any_bbox(bboxes: list, obj: dict) -> bool:
    """判断字符对象是否完全落在某个表格区域内。"""
    for x0, top, x1, bottom in bboxes:
        if (
            obj["x0"] >= x0 - 1
            and obj["top"] >= top - 1
            and obj["x1"] <= x1 + 1
            and obj["bottom"] <= bottom + 1
        ):
            return True
    return False


def extract_page_content(page, *, preserve_unrepresented_text: bool = True) -> tuple[str, list]:
    """仅在表格完整保留区域文字时剔除正文；旧策略可用于消融对照。"""
    text = ""
    tables: list = []
    try:
        found = page.find_tables()
        tables = page.extract_tables() or []
        if preserve_unrepresented_text:
            verified_tables, verified_boxes = [], []
            # 检测区域可能包含未落入任何单元格的文字。按同一个过滤边界核对
            # 非空白字符及其数量，不能仅凭区域框就从正文删除它们。
            if len(found) == len(tables):
                for detected, table in zip(found, tables):
                    region = page.filter(lambda obj: _inside_any_bbox([detected.bbox], obj))
                    region_chars = Counter(c for c in (region.extract_text() or '') if not c.isspace())
                    cell_chars = Counter(c for row in table for cell in row
                                         for c in str(cell or '') if not c.isspace())
                    if region_chars and not (region_chars - cell_chars):
                        verified_tables.append(table)
                        verified_boxes.append(detected.bbox)
            tables = verified_tables
            bboxes = verified_boxes
        else:
            bboxes = [t.bbox for t in found]
        if bboxes:
            body = page.filter(lambda obj: not _inside_any_bbox(bboxes, obj))
            text = body.extract_text() or ""
        else:
            text = page.extract_text() or ""
    except Exception as exc:
        print(f"[警告] 第 {page.page_number} 页表格/文本提取失败: {exc}", file=sys.stderr)
        # 表格处理失败时仍尝试保留整页原文，不返回空页或重复表格。
        text = page.extract_text() or ""
        tables = []
    return text, tables


def collect_pages(pdf, *, preserve_unrepresented_text: bool = True) -> list[dict]:
    """逐页提取正文文本（剔除表格区域）与表格。"""
    pages = []
    for i, page in enumerate(pdf.pages, start=1):
        try:
            text, tables = extract_page_content(page, preserve_unrepresented_text=preserve_unrepresented_text)
        except Exception as exc:  # 单页失败不阻断整体
            text, tables = "", []
            print(f"[警告] 第 {i} 页提取失败: {exc}", file=sys.stderr)
        pages.append(
            {
                "page": i,
                "text_raw": text,
                "tables": tables,
                "has_image": bool(getattr(page, "images", None)),
            }
        )
    return pages


def strip_frame_noise(pages: list[dict]) -> tuple[list[dict], int]:
    """剔除页眉/页脚噪声行。

    规则：某行规范化后出现在 >= NOISE_MIN_PAGE_FREQ 个不同页，且位于该页
    首/尾 NOISE_MAX_EDGE 行内 -> 判定为跨页框架文本（册名/页码等）并剔除。
    """
    freq: Counter = Counter()
    for p in pages:
        seen = {_norm(line) for line in p["text_raw"].splitlines()}
        freq.update(seen)
    removed = 0
    for p in pages:
        lines = p["text_raw"].splitlines()
        n = len(lines)
        keep: list[str] = []
        for idx, raw in enumerate(lines):
            ln = _norm(raw)
            if not ln:
                continue
            near_edge = idx < NOISE_MAX_EDGE or idx >= n - NOISE_MAX_EDGE
            is_page_no = ln.isdigit() and idx >= n - NOISE_MAX_EDGE  # 页尾孤立页码
            if (near_edge and freq[ln] >= NOISE_MIN_PAGE_FREQ) or is_page_no:
                removed += 1
                continue
            keep.append(raw.rstrip())
        p["text"] = "\n".join(keep)
    return pages, removed


def page_markdown(p: dict) -> str:
    """单页正文 + 表格 -> Markdown 片段。"""
    parts = [p["text"].strip()]
    for table in p["tables"]:
        md = table_to_markdown(table)
        if md:
            parts.append(md)
    return "\n\n".join(part for part in parts if part.strip())


def annotate(pages: list[dict]) -> list[dict]:
    """标记封面 / 目录 / 空页，计算清洗后字符数。"""
    for idx, p in enumerate(pages):
        p["is_cover"] = idx == 0 and len(p["text"].strip()) < 60
        p["is_toc"] = (not p["is_cover"]) and _is_toc_page(p["text"])
        p["char_len"] = len(p["text"].strip())
        p.pop("text_raw", None)
    return pages


def main() -> int:
    ensure_utf8()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PDF_PATH.exists():
        print(f"[错误] 找不到 PDF: {PDF_PATH}", file=sys.stderr)
        return 1

    print(f"[1/5] 打开 {PDF_PATH.name} ...")
    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
        print(f"[2/5] 提取 {total_pages} 页文本与表格 ...")
        raw_pages = collect_pages(pdf)
        raw_char_sum = sum(len(p.get("text_raw") or "") for p in raw_pages)

    print("[3/5] 剔除页眉/页脚噪声 ...")
    pages, removed_lines = strip_frame_noise(raw_pages)
    pages = annotate(pages)

    # ---- 写出 handbook_editable.md ----
    print("[4/5] 写出可编辑 Markdown ...")
    md_parts = [
        "# 京东集团员工手册 —— 可编辑文本版",
        "",
        f"> 由 ragcore/extract.py 从 PDF 自动提取生成 · 共 {total_pages} 页 · "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> 每段正文前有 `<!-- page N [cover|toc] -->` 页标记，便于溯源与后续分块。",
        "",
    ]
    for p in pages:
        tag = "cover" if p["is_cover"] else ("toc" if p["is_toc"] else "")
        md_parts.append(f"<!-- page {p['page']}{(' ' + tag) if tag else ''} -->")
        body = page_markdown(p)
        if body:
            md_parts.append(body)
        md_parts.append("")
    EDITABLE_MD.write_text("\n".join(md_parts), encoding="utf-8")

    # ---- 写出 pages.jsonl ----
    with PAGES_JSONL.open("w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ---- 统计 ----
    print("[5/5] 汇总统计 ...")
    empty = [p["page"] for p in pages if p["char_len"] == 0 and not p["tables"] and not p["is_cover"]]
    toc_pages = [p["page"] for p in pages if p["is_toc"]]
    table_pages = [p["page"] for p in pages if p["tables"]]
    stats = {
        "step": "extract",
        "source": str(PDF_PATH),
        "total_pages": total_pages,
        "raw_chars": raw_char_sum,
        "clean_chars": sum(p["char_len"] for p in pages),
        "empty_pages": empty,
        "toc_pages": toc_pages,
        "cover_pages": [p["page"] for p in pages if p["is_cover"]],
        "table_pages": table_pages,
        "pages_with_images": [p["page"] for p in pages if p["has_image"]],
        "noise_lines_removed": removed_lines,
        "editable_md_bytes": EDITABLE_MD.stat().st_size,
    }
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 控制台汇总 ----
    print("-" * 60)
    print(f"总页数           : {total_pages}")
    print(f"原始文本字符     : {stats['raw_chars']:,}   (清洗后 {stats['clean_chars']:,})")
    print(f"剔除噪声行       : {removed_lines}")
    print(f"空页             : {empty or '无'}")
    print(f"目录页           : {toc_pages or '无'}")
    print(f"含表格页         : {table_pages or '无'}")
    print(f"含图片页         : {stats['pages_with_images'] or '无'}")
    print(f"产物             : {EDITABLE_MD} ({stats['editable_md_bytes']:,} B)")
    print(f"                  {PAGES_JSONL}")
    print(f"                  {STATS_JSON}")
    print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

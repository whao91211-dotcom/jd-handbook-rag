"""Step 2：结构识别 + 分块。

输入：data/pages.jsonl（Step 1 产物）
输出：data/chunks.jsonl  每块：
  {
    id, pages, chapter, section, clause_range, path,
    content(不带前缀), text(带【章节路径】前缀), char_len
  }
  data/stats_chunk.json

分块策略：
  1) 章/节标题 -> 结构上下文（必切）；
  2) 编号条款：块内容 >= 目标*0.6 时切块重开；否则并入当前块并保留条款标题行
     （把同一节下相邻的短条款合并到 ~目标长度，减少碎块）；
  3) 超长连续文本按句子窗口切（重叠 CHUNK_OVERLAP）。

用法：& "...\\InternVL\\python.exe" -m ragcore.chunker
"""
from __future__ import annotations

import hashlib
import json
import re
import sys

from .config import (
    CHUNK_MAX,
    CHUNK_OVERLAP,
    CHUNK_TARGET,
    CHUNKS_JSONL,
    PAGES_JSONL,
    STATS_CHUNK_JSON,
)

RE_CHAPTER = re.compile(r"^第([一二三四五六七八九十百零]+)章\s*(.*)$")
RE_SECTION = re.compile(r"^第([一二三四五六七八九十百零]+)节\s*(.*)$")
RE_CLAUSE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\s*(.*)$")   # 至少两级编号，如 12.1 / 5.2.3

MERGE_MIN = int(CHUNK_TARGET * 0.6)   # 块达到该长度后，新条款到来即切块


def ensure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def load_pages() -> list[dict]:
    with PAGES_JSONL.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------- 长文本按句子窗口切分（带重叠） ----------------

def _sentence_bounds(text: str, lo: int, hi: int) -> int:
    for delim in ("。", "；", "！", "？"):
        pos = text.rfind(delim, lo, hi)
        if pos != -1:
            return pos + 1
    pos = text.rfind("\n", lo, hi)
    if pos != -1:
        return pos + 1
    return hi


def split_with_overlap(text: str, target: int, max_len: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    out: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_len, n)
        if end - start > target:
            end = _sentence_bounds(text, start + target, end)
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        nxt = end - overlap
        start = nxt if nxt > start else end
    return out


# ---------------- 表格分块 ----------------

def _md_line(cells: list, ncols: int) -> str:
    cells = (list(cells) + [""] * ncols)[:ncols]
    return "| " + " | ".join(str(c).strip() for c in cells) + " |"


def split_table(table: list[list], max_len: int) -> list[str]:
    if not table:
        return []
    rows = [r for r in table if any(c not in (None, "") for c in r)]
    if not rows:
        return []
    header = ["" if c is None else str(c).replace("\n", " ") for c in rows[0]]
    data = [[("" if c is None else str(c).replace("\n", " ").replace("|", "\\|")) for c in r] for r in rows[1:]]
    ncols = max(len(header), *(len(r) for r in data))
    one = "\n".join([_md_line(header, ncols), _md_line(["---"] * ncols, ncols)] + [_md_line(r, ncols) for r in data])
    if len(one) <= max_len:
        return [one]
    group = max(4, max_len // (max(len(line) for line in one.splitlines()) + 1))
    pieces = []
    for i in range(0, len(data), group):
        chunk = "\n".join([_md_line(header, ncols), _md_line(["---"] * ncols, ncols)]
                          + [_md_line(r, ncols) for r in data[i : i + group]])
        pieces.append(chunk)
    return pieces


# ---------------- 主体：流式累积 ----------------

class ChunkBuilder:
    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.chapter = ""
        self.section = ""
        self.block: list[str] = []
        self.block_pages: list[int] = []
        self.first_clause: str | None = None
        self.last_clause: str | None = None
        self.last_clause_title = ""

    # ---- 状态 ----
    def block_len(self) -> int:
        return sum(len(x) for x in self.block) + len(self.block)

    def block_path(self) -> str:
        parts = [p for p in (self.chapter, self.section) if p]
        if self.first_clause:
            if self.last_clause and self.last_clause != self.first_clause:
                parts.append(f"{self.first_clause}~{self.last_clause}")
            else:
                parts.append(f"{self.first_clause} {self.last_clause_title}".strip())
        return " > ".join(parts)

    def _make_id(self, text: str, page: int) -> str:
        return f"p{page}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"

    # ---- 块操作 ----
    def flush(self) -> None:
        content = "\n".join(self.block).strip()
        if content:
            if len(content) > CHUNK_MAX:
                for piece in split_with_overlap(content, CHUNK_TARGET, CHUNK_MAX, CHUNK_OVERLAP):
                    self._emit(piece)
            else:
                self._emit(content)
        self.block = []
        self.block_pages = []
        self.first_clause = None
        self.last_clause = None
        self.last_clause_title = ""

    def _emit(self, content: str) -> None:
        content = content.strip()
        if not content:
            return
        pages = sorted(set(self.block_pages)) if self.block_pages else []
        if not pages:
            return
        path = self.block_path()
        text = f"【{path}】\n{content}" if path else content
        self.chunks.append(
            {
                "id": self._make_id(text, pages[0]),
                "pages": pages,
                "chapter": self.chapter,
                "section": self.section,
                "clause_range": f"{self.first_clause}~{self.last_clause}" if self.first_clause and self.last_clause != self.first_clause else (self.first_clause or ""),
                "path": path,
                "content": content,
                "text": text,
                "char_len": len(content),
            }
        )

    # ---- 行处理 ----
    def add_line(self, line: str, page: int) -> None:
        line = line.strip()
        if not line:
            return
        m = RE_CHAPTER.match(line)
        if m:
            self.flush()
            self.chapter = line
            self.section = ""
            return
        m = RE_SECTION.match(line)
        if m:
            self.flush()
            self.section = line
            return
        m = RE_CLAUSE.match(line)
        if m:
            clause_id, rest = m.group(1), m.group(2).strip()
            if self.block and self.block_len() >= MERGE_MIN:
                self.flush()
            if self.first_clause is None:
                self.first_clause = clause_id
            self.last_clause = clause_id
            self.last_clause_title = rest
            if rest and len(rest) >= 30:      # 标题行同时携带正文
                self._push(rest, page)
            elif rest:                        # 纯标题行：保留为块内可见小标题
                self._push(f"{clause_id} {rest}", page)
            else:
                self._push(clause_id, page)
            return
        self._push(line, page)

    def _push(self, line: str, page: int) -> None:
        if self.block and self.block_len() + len(line) > CHUNK_MAX:
            self.flush()
        if page not in self.block_pages:
            self.block_pages.append(page)
        self.block.append(line)


def build_chunks(pages: list[dict]) -> list[dict]:
    b = ChunkBuilder()
    for p in pages:
        if p.get("is_cover") or p.get("is_toc"):
            continue
        page = int(p["page"])
        for line in p["text"].splitlines():
            b.add_line(line, page)
        # 该页表格：按当前上下文成块（表体已从正文剔除，不会重复）
        for table in p["tables"]:
            b.flush()
            for piece in split_table(table, CHUNK_MAX):
                b.block = [piece]
                b.block_pages = [page]
                b.flush()
    b.flush()
    return b.chunks


def main() -> int:
    ensure_utf8()
    pages = load_pages()
    print(f"[1/3] 读取 {len(pages)} 页记录 ...")
    chunks = build_chunks(pages)
    print(f"[2/3] 生成 {len(chunks)} 个分块 ...")

    total_clean = sum(p["char_len"] for p in pages if not p.get("is_cover") and not p.get("is_toc"))
    covered = sum(c["char_len"] for c in chunks)
    lens = [c["char_len"] for c in chunks]
    dup_text = len(chunks) - len({c["content"] for c in chunks})

    with CHUNKS_JSONL.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    stats = {
        "step": "chunk",
        "n_chunks": len(chunks),
        "content_chars": covered,
        "source_clean_chars": total_clean,
        "coverage_ratio": round(covered / total_clean, 4) if total_clean else None,
        "len_min": min(lens) if lens else 0,
        "len_max": max(lens) if lens else 0,
        "len_avg": round(sum(lens) / len(lens), 1) if lens else 0,
        "dup_content": dup_text,
        "chapters": sorted({c["chapter"] for c in chunks if c["chapter"]}),
        "with_clause": sum(1 for c in chunks if c["clause_range"]),
        "with_page_span": sum(1 for c in chunks if len(c["pages"]) > 1),
    }
    STATS_CHUNK_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 60)
    print(f"分块数          : {stats['n_chunks']}")
    print(f"覆盖字符        : {covered:,} / {total_clean:,}  ({stats['coverage_ratio'] * 100:.1f}%)")
    print(f"块长 min/avg/max: {stats['len_min']} / {stats['len_avg']} / {stats['len_max']}")
    print(f"重复正文块      : {dup_text}")
    print(f"带条款编号块    : {stats['with_clause']} ({stats['with_clause'] / max(stats['n_chunks'], 1) * 100:.0f}%)")
    print(f"跨页块          : {stats['with_page_span']}")
    print(f"识别章节({len(stats['chapters'])})  : " + " | ".join(stats["chapters"]))
    print(f"产物            : {CHUNKS_JSONL}")
    print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

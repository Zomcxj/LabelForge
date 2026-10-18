# -*- coding: utf-8 -*-
"""词典里同一个键不能写两遍。

写两遍时 Python 不报错，后一条静默覆盖前一条 —— 结果是某些词条永远翻译不
出来，而码里看不出任何异常。曾经就有 `空` 被 `Emp` 和 `Empty` 各写一次，
`输出目录` 被写两次（Output folder / Output dir），后写的悄悄赢了。
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I18N = ROOT / "labelforge" / "ui" / "i18n.py"


def _dict_literals(tree):
    """取出文件里所有 dict display 节点（翻译表就是一堆 dict 字面量）。"""
    return [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]


def test_no_duplicate_keys_in_any_dict():
    """同一 dict 里键不能重复；同值的自我映射重复除外。

    中文词典里有少量「键与值相同」的重复（历史遗留），它们不影响行为，
    属于可以慢慢清理的噪音；但**值不同**的重复一定会静默覆盖，必须为零。
    """
    tree = ast.parse(I18N.read_text(encoding="utf-8"))
    problems = []
    for node in _dict_literals(tree):
        seen = {}
        for key_node, value_node in zip(node.keys, node.values):
            if not isinstance(key_node, ast.Constant):
                continue
            key = key_node.value
            if key not in seen:
                seen[key] = (key_node, value_node)
                continue
            prev_key, prev_value = seen[key]
            try:
                prev = ast.literal_eval(prev_value)
                cur = ast.literal_eval(value_node)
            except Exception:
                prev, cur = "?", "?"
            if prev == cur:
                continue          # 同值重复：无害，不在此处拦
            problems.append(
                f"键 {key!r} 在第 {prev_key.lineno} 行与第 {key_node.lineno} 行"
                f"各写了一次，值不同（{prev!r} 会被 {cur!r} 覆盖）")
    assert not problems, "\n".join(problems)


def test_english_dict_translates_the_punctuated_labels():
    """带冒号的标签要真有英文，别在清理重复键时被顺手删掉。"""
    text = I18N.read_text(encoding="utf-8")
    for key in ("基本颜色：", "自定义颜色："):
        pattern = rf'"{re.escape(key)}":\s*"([^"]+)"'
        values = re.findall(pattern, text)
        assert values, f"找不到 {key} 的翻译"
        # 至少有一条不是原样返回中文（即真的翻成了英文）
        assert any(v != key for v in values), (
            f"{key} 只有中文原样返回，英文词典里缺这条")

"""把源码编译出的字节码与归档（exe 内嵌 PYZ）里的字节码逐指令比对。

用途：确认「打包产物真的就是这份源码」，而不是某个陈旧缓存。

三类**无语义**差异必须剔除，否则会把「一致」误判成「不一致」（上一轮踩过）：
  1. PyInstaller 用 optimize=2 打包，会剥掉模块文档字符串。
  2. loader 会在模块头插入 `NOP`。
  3. 跳转指令的字节偏移整体平移（前面少了指令，偏移自然不同）。
另外 `RESUME` / `CACHE` / `EXTENDED_ARG` 在不同 Python 版本/优化级别下
存在与否不同，也一并忽略。

归档格式不自己解析：用 PyInstaller 自己的 `CArchiveReader`，免得跟着它改格式。

用法：
  python tools/compare_archived_bytecode.py <exe路径> [模块名...]
"""
import dis
import io
import marshal
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 这些指令只影响「优化级别/解释器实现」，不表达语义
IGNORED = {"RESUME", "CACHE", "EXTENDED_ARG", "NOP"}


def _const_key(value):
    """把常量转成**稳定**的可比较键。

    直接比 `argrepr` 会踩集合顺序：`frozenset({'w','overlap','h'})` 与
    `frozenset({'overlap','h','w'})` 是同一个值，但 repr 顺序随哈希种子变，
    跨进程不稳定——会把一模一样的代码判成不一致（踩过，且只在带 frozenset
    常量的模块上偶发，最难查）。
    """
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(v) for v in value)))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((repr(k), repr(v)) for k, v in value.items())))
    if hasattr(value, "co_name"):        # 嵌套 code object
        return ("code", value.co_name)
    if isinstance(value, float) and value != value:   # NaN
        return ("float", "nan")
    try:
        hash(value)
    except TypeError:
        return ("repr", repr(value))
    return ("val", value)


def _flatten(code):
    """把 code object 展平成可比对的指令序列（剔除无语义差异）。"""
    out = []
    queue = [code]
    seen = set()
    while queue:
        co = queue.pop(0)
        if id(co) in seen:
            continue
        seen.add(id(co))
        out.append(("CODE", co.co_name, co.co_argcount, co.co_varnames))
        nested = []
        for ins in dis.get_instructions(co):
            if ins.opname in IGNORED:
                continue
            # 跳转目标用「跳转」占位而不是字节偏移：偏移会整体平移
            if ins.opcode in dis.hasjabs or ins.opcode in dis.hasjrel:
                out.append((ins.opname, "<jump>"))
                continue
            if ins.opname in ("LOAD_CONST", "RETURN_CONST"):
                out.append((ins.opname, _const_key(ins.argval)))
                continue
            out.append((ins.opname, ins.argrepr))
        for const in co.co_consts:
            if isinstance(const, type(co)):
                nested.append(const)
        queue.extend(nested)
    return out


def _strip_doc(code):
    """历史遗留：现在不需要手工剥文档串了。

    `source_code()` 用 `optimize=2` 编译，Python 自己就会把文档字符串剥掉、
    连同那两条 `LOAD_CONST`/`STORE_NAME __doc__` 指令一起，与归档口径完全
    一致（归档就是 optimize=2 打出来的）。手工近似剥除只会引入错位：
    类体的 `co_consts[0]` 是 `__qualname__`，函数文档串位置的启发式判据
    分不清「类的文档串」与「类体里第一个字符串常量」（枚举值就会误判），
    而任何一处误判都会让之后所有指令错位、把每个模块都判成不一致。

    保留这个函数名只为兼容旧调用点，行为是恒等。
    """
    return code


def source_code(module_name):
    """从源码编译出 code object（optimize=2，与归档对齐）。"""
    path = ROOT / (module_name.replace(".", "/") + ".py")
    if not path.exists():
        path = ROOT / module_name.replace(".", "/") / "__init__.py"
    src = path.read_text(encoding="utf-8")
    return compile(src, str(path), "exec", optimize=2)


def archived_codes(exe_path):
    """从 exe 的 PYZ 里取出 {模块名: code object}。"""
    from PyInstaller.archive.readers import CArchiveReader

    arch = CArchiveReader(str(exe_path))
    result = {}
    for name in arch.toc:
        # PYZ 条目名形如 'PYZ.pyz' / 'pyz-00.pyz'
        if not name.lower().endswith(".pyz"):
            continue
        raw = arch.extract(name)
        if raw is None:
            continue
        # pyz 本身又是 PyInstaller 的 ZlibArchive
        import zlib
        from PyInstaller.archive.readers import ZlibArchiveReader

        tmp = io.BytesIO(raw)
        pyz = ZlibArchiveReader.__new__(ZlibArchiveReader)
        # 直接复用它的解析：先写出临时文件再读，最省事也最稳
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pyz", delete=False) as fh:
            fh.write(raw)
            tmp_name = fh.name
        try:
            pyz = ZlibArchiveReader(tmp_name)
            for mod in pyz.toc:
                if mod.startswith("_"):
                    continue
                try:
                    code = pyz.extract(mod)
                except Exception:
                    continue
                if isinstance(code, type(compile("", "<x>", "exec"))):
                    result[mod] = code
        finally:
            pathlib.Path(tmp_name).unlink(missing_ok=True)
    return result


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    exe = argv[1]
    wanted = set(argv[2:]) or None

    arch = archived_codes(exe)
    print(f"归档里解析出 {len(arch)} 个模块")
    if not arch:
        print("没解析出任何模块 —— 请确认这是 PyInstaller 产物")
        return 1

    mismatched, skipped, checked = [], [], 0
    for mod, code in sorted(arch.items()):
        if not mod.startswith("labelforge"):
            continue
        if wanted and mod not in wanted:
            continue
        if not (ROOT / (mod.replace(".", "/") + ".py")).exists():
            skipped.append(mod)
            continue
        checked += 1
        try:
            src_code = source_code(mod)
        except Exception as exc:
            mismatched.append(f"{mod}: 源码编译失败 {exc}")
            continue
        # 两侧都用同一套归一化；`optimize=2` 已让源码侧与归档侧同为「无文档串」
        if _flatten(src_code) != _flatten(code):
            mismatched.append(mod)

    print(f"比对 {checked} 个 labelforge 模块"
          + (f"，跳过 {len(skipped)} 个无源码" if skipped else ""))
    if mismatched:
        print(f"!! {len(mismatched)} 个模块不一致:")
        for mod in mismatched:
            print(f"  - {mod}")
        return 1
    print("全部一致（已剔除文档字符串/NOP/跳转偏移三类无语义差异）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

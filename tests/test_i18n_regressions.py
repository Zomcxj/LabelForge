"""国际化回归测试。"""
import string

from labelforge.ui import i18n


def test_iou_axis_label_has_explicit_translation():
    """轴标签走 tr('IoU')，两个语言表都要有该键（否则回退到键名，语言表不完整）。"""
    assert 'IoU' in i18n._strings['zh']
    assert 'IoU' in i18n._strings['en']


def test_english_translates_magnifier_and_detection_box_wheel_settings():
    original_lang = i18n.get_lang()
    try:
        i18n.set_lang('en')
        assert i18n.t('窗口放大器倍率') == 'Magnifier Zoom'
        assert i18n.t('检测框缩放步长') == 'Box Wheel Scale Step'
        assert i18n.t('单侧位移像素') == 'Box Edge Nudge Pixels'
    finally:
        i18n.set_lang(original_lang)


def test_crop_template_strings_exist_in_both_languages():
    for key in ('模板', 'template_fmt'):
        assert key in i18n._strings['zh'], key
        assert key in i18n._strings['en'], key


def test_crop_template_format_fields_match_across_languages():
    def fields(s):
        return [f for _, f, _, _ in string.Formatter().parse(s) if f]
    zh_fields = fields(i18n._strings['zh']['template_fmt'])
    en_fields = fields(i18n._strings['en']['template_fmt'])
    assert zh_fields == en_fields == ['w', 'h', 'ov', 'vis']


def test_crop_template_format_renders_in_both_languages():
    original_lang = i18n.get_lang()
    try:
        i18n.set_lang('zh')
        assert i18n.t('模板') == '模板'
        assert i18n.t('template_fmt').format(
            w=800, h=640, ov=200, vis=50) == '800×640·重叠200·50%'
        i18n.set_lang('en')
        assert i18n.t('模板') == 'Templates'
        assert i18n.t('template_fmt').format(
            w=800, h=640, ov=200, vis=50) == '800×640·overlap 200·50%'
    finally:
        i18n.set_lang(original_lang)


def _iter_tr_keys():
    """扫 labelforge/ 下所有 `tr("...")` / `t("...")` 的字面量 key。"""
    import ast
    import os

    root_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'labelforge')
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for name in files:
            if not name.endswith('.py') or name == 'i18n.py':
                continue
            path = os.path.join(root, name)
            with open(path, encoding='utf-8') as fh:
                try:
                    tree = ast.parse(fh.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, 'id', '') not in ('tr', 't'):
                    continue
                if not node.args:
                    continue
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield arg.value, f"{os.path.relpath(path, root_dir)}:{node.lineno}"


def test_every_tr_key_exists_in_both_language_tables():
    """每个 `tr()` 的 key 都必须在**两张**表里。

    用户报的「设置里显示标签名/恢复图片/窗口放大器尺寸/放大器显示位置/
    标签名位置没有同步英文」，根因不是 en 漏了，而是 **zh 里也没有这个 key**——
    `t()` 是 `_strings[lang].get(key, key)`，回落成 key 本身；key 又正好是中文，
    于是中文界面看着正常、英文界面照样显示中文，光看中文界面永远发现不了。

    所以断言两张表都要有该键，而不是只比对 zh/en 差集。
    """
    missing = {}
    for key, where in _iter_tr_keys():
        for lang in ('zh', 'en'):
            if key not in i18n._strings[lang]:
                missing.setdefault(key, set()).add(lang)
    assert not missing, (
        "以下 tr() 的 key 至少缺一张语言表（会回落成 key 本身，"
        "key 是中文时英文界面就显示中文）：\n"
        + "\n".join(f"  {k!r} 缺 {sorted(v)}" for k, v in sorted(missing.items()))
    )


def test_language_tables_do_not_diverge_on_tr_keys():
    """两张表在 **tr() 真正用到的 key** 上必须一致。

    只比对整张表的差集是**误报**：有些键的 canonical 形式就是中文（例如增强
    操作的名称 `tr("亮度")`、`tr("高斯噪声")`，中文表靠「key 即译文」回落，
    所以只在 en 表里出现）。真正要保证的是「代码里 tr() 用到的每个 key 两张
    表都有」，那是上一个测试。这里只补一条：两张表都声明的键，不能只声明一半。
    """
    zh = set(i18n._strings['zh'])
    en = set(i18n._strings['en'])
    used = {key for key, _ in _iter_tr_keys()}
    only_zh = sorted((zh - en) & used)
    only_en = sorted((en - zh) & used)
    assert not only_zh and not only_en, (
        f"只在 zh: {only_zh}\n只在 en: {only_en}")

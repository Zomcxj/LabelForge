"""国际化回归测试。"""
import string

from pastelabel.ui import i18n


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

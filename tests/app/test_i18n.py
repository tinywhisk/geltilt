from __future__ import annotations

from pathlib import Path

from geltilt.app import config as app_config
from geltilt.app import i18n, settings


def test_default_language_is_simplified_chinese():
    assert i18n.DEFAULT_LANGUAGE == "zh"
    assert set(i18n.available_languages()) == {"zh", "en"}
    assert i18n.language_label("zh") == "\u7b80\u4f53\u4e2d\u6587"


def test_tr_returns_chinese_for_default_keys():
    i18n.set_language("zh")
    assert i18n.tr("btn.auto_rotate") == "\u81ea\u52a8\u65cb\u8f6c"
    assert i18n.tr("diag.accepted") == "\u5df2\u63a5\u53d7"


def test_tr_english_and_chinese_differ():
    en = i18n.tr("btn.save_as", language="en")
    zh = i18n.tr("btn.save_as", language="zh")
    assert en == "Save as..."
    assert zh != en


def test_tr_formats_placeholders():
    i18n.set_language("en")
    assert i18n.tr("status.samples_in", count=3, name="gels") == "3 sample(s) in gels"
    i18n.set_language("zh")
    assert "3" in i18n.tr("status.samples_in", count=3, name="gels")


def test_unknown_key_returns_key_and_unknown_language_falls_back():
    i18n.set_language("zh")
    assert i18n.tr("no.such.key") == "no.such.key"
    assert i18n.set_language("klingon") == i18n.DEFAULT_LANGUAGE


def test_language_label_round_trips():
    for code in i18n.available_languages():
        assert i18n.language_code(i18n.language_label(code)) == code


def test_diagnostic_reason_is_localized_only_in_chinese():
    text = "ok: 2 lane(s) pooled of 3, 12 rungs, se 0.30 deg"
    i18n.set_language("en")
    assert i18n.localize_diagnostic(text) == text
    i18n.set_language("zh")
    localized = i18n.localize_diagnostic(text)
    assert localized != text and "0.30" in localized
    assert i18n.localize_diagnostic("some unknown wording") == "some unknown wording"


def test_all_catalogued_keys_have_both_languages():
    from geltilt.app.i18n import _STRINGS

    for key, entry in _STRINGS.items():
        assert set(entry) == {"en", "zh"}, key
        assert entry["en"] and entry["zh"], key


def test_settings_round_trip(tmp_path: Path):
    assert settings.load_language(tmp_path) == i18n.DEFAULT_LANGUAGE
    assert settings.save_language(tmp_path, "en") is True
    assert settings.load_language(tmp_path) == "en"
    assert settings.save_language(tmp_path, "nope") is False
    assert settings.load_language(tmp_path) == "en"


def test_app_config_reads_persisted_language(tmp_path: Path, monkeypatch):
    settings.save_language(tmp_path, "en")
    monkeypatch.setattr(app_config, "load_language", settings.load_language)
    conf = app_config.AppConfig.resolve(run_dir=tmp_path)
    assert conf.language == "en"


def test_app_config_defaults_to_chinese_when_unset(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(app_config, "load_language", settings.load_language)
    conf = app_config.AppConfig.resolve(run_dir=tmp_path)
    assert conf.language == "zh"
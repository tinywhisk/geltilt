from __future__ import annotations

from geltilt.app.about import bundle_root, licenses_text


def test_bundle_root_has_the_license_files():
    assert (bundle_root() / "LICENSE").is_file()
    assert (bundle_root() / "THIRD_PARTY_NOTICES.md").is_file()


def test_licenses_text_has_app_license_and_third_party_notices():
    text = licenses_text()
    assert "MIT License" in text          # our own license
    assert "Third-Party Notices" in text  # the notices heading
    assert "Apache-2.0" in text           # OpenCV / GelGenie
    assert "PyInstaller" in text
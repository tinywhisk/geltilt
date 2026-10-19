"""Tiny dependency-free localization layer for the desktop app.

The app ships two languages: Simplified Chinese (``"zh"``, the default) and
English (``"en"``).  Every user-facing string is looked up through :func:`tr`
using a stable key; the active language is a module global set once at launch by
:class:`~geltilt.app.window.AppWindow` (from the persisted setting) and never
changed while the window is live -- the toolbar combobox only *saves* the
selection and asks the user to restart (see ``window.py``).

Strings are plain ``str.format`` templates; because of that any literal brace in
a message must be doubled (``{{`` / ``}}``), matching the lane-spec messages the
controller already used.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_LANGUAGE = "zh"

# Native display names, in the order shown in the toolbar combobox.
LANGUAGES: Dict[str, str] = {
    "zh": "\u7b80\u4f53\u4e2d\u6587",  # Simplified Chinese
    "en": "English",
}

_LANG = DEFAULT_LANGUAGE

_STRINGS: Dict[str, Dict[str, str]] = {
    # -- window chrome ----------------------------------------------------
    "app.title": {"en": "geltilt - gel rotation", "zh": "geltilt - \u51dd\u80f6\u65cb\u8f6c"},
    "btn.load_image": {"en": "Load image...", "zh": "\u52a0\u8f7d\u56fe\u50cf\u2026"},
    "btn.load_folder": {"en": "Load folder...", "zh": "\u52a0\u8f7d\u6587\u4ef6\u5939\u2026"},
    "label.language": {"en": "Language:", "zh": "\u8bed\u8a00\uff1a"},
    "group.rotation": {"en": " Rotation ", "zh": " \u65cb\u8f6c "},
    "group.crop": {"en": " Crop ", "zh": " \u88c1\u526a "},
    "group.export": {"en": " Export ", "zh": " \u5bfc\u51fa "},
    "group.overlays": {"en": " Overlays ", "zh": " \u53e0\u52a0\u5c42 "},
    "label.corrected": {"en": "Corrected", "zh": "\u6821\u6b63\u540e"},
    "btn.auto_rotate": {"en": "Auto rotate", "zh": "\u81ea\u52a8\u65cb\u8f6c"},
    "btn.auto_crop": {"en": "Auto crop", "zh": "\u81ea\u52a8\u88c1\u526a"},
    "btn.clear_crop": {"en": "Clear crop", "zh": "\u6e05\u9664\u88c1\u526a"},
    "btn.even_widths": {"en": "Even widths", "zh": "\u7b49\u5bbd"},
    "btn.save_as": {"en": "Save as...", "zh": "\u53e6\u5b58\u4e3a\u2026"},
    "btn.copy_clipboard": {"en": "Copy to clipboard", "zh": "\u590d\u5236\u5230\u526a\u8d34\u677f"},
    "btn.about": {"en": "About / Licenses", "zh": "\u5173\u4e8e / \u8bb8\u53ef\u8bc1"},
    "btn.reset": {"en": "Reset", "zh": "\u91cd\u7f6e"},
    "btn.close": {"en": "Close", "zh": "\u5173\u95ed"},

    # -- rotation / crop / export controls --------------------------------
    "label.manual_angle": {"en": "Manual angle (\u00b0):", "zh": "\u624b\u52a8\u89d2\u5ea6 (\u00b0)\uff1a"},
    "label.no_crop": {"en": "no crop", "zh": "\u672a\u88c1\u526a"},
    "label.padding": {"en": "Padding:", "zh": "\u8fb9\u8ddd\uff1a"},
    "label.padding_unit": {"en": "px  left/right/bottom", "zh": "\u50cf\u7d20\uff08\u5de6/\u53f3/\u4e0b\uff09"},
    "check.composite_crop": {"en": "Composite crop", "zh": "\u590d\u5408\u88c1\u526a"},
    "hint.run_auto_crop_first": {"en": "run Auto crop first",
                                 "zh": "\u8bf7\u5148\u6267\u884c\u81ea\u52a8\u88c1\u526a"},
    "hint.composite_click": {"en": "click lanes to join, or type e.g. 3,1-2,4-6",
                             "zh": "\u70b9\u51fb\u6cf3\u9053\u4ee5\u5408\u5e76\uff0c\u6216\u8f93\u5165\u5982 3,1-2,4-6"},
    "hint.composite_tick": {"en": "tick to join lanes; click a lane to add it",
                            "zh": "\u52fe\u9009\u4ee5\u5408\u5e76\u6cf3\u9053\uff1b\u70b9\u51fb\u6cf3\u9053\u4ee5\u6dfb\u52a0"},
    "label.save_copy": {"en": "Save / copy:", "zh": "\u4fdd\u5b58 / \u590d\u5236\uff1a"},
    "export.mode_rotation_only": {"en": "Rotation only", "zh": "\u4ec5\u65cb\u8f6c"},
    "export.mode_rotation_crop": {"en": "Rotation + crop", "zh": "\u65cb\u8f6c + \u88c1\u526a"},
    "check.use_original": {"en": "Use original image (no contrast/brightness)",
                           "zh": "\u4f7f\u7528\u539f\u59cb\u56fe\u50cf\uff08\u4e0d\u8c03\u6574\u5bf9\u6bd4\u5ea6/\u4eae\u5ea6\uff09"},
    "label.8bit": {"en": "8-bit", "zh": "8 \u4f4d"},
    "check.vertical_guides": {"en": "Vertical guides", "zh": "\u5782\u76f4\u53c2\u8003\u7ebf"},
    "check.band_centroids": {"en": "Band centroids", "zh": "\u6761\u5e26\u8d28\u5fc3"},
    "check.lane_lines": {"en": "Accepted lane lines", "zh": "\u5df2\u63a5\u53d7\u7684\u6cf3\u9053\u7ebf"},
    "check.gelgenie_masks": {"en": "GelGenie masks", "zh": "GelGenie \u63a9\u819c"},

    # -- diagnostics panel -------------------------------------------------
    "group.diagnostics": {"en": "Diagnostics", "zh": "\u8bca\u65ad"},
    "diag.sample": {"en": "Sample", "zh": "\u6837\u672c"},
    "diag.applied": {"en": "Applied", "zh": "\u5df2\u5e94\u7528"},
    "diag.status": {"en": "Status", "zh": "\u72b6\u6001"},
    "diag.uncertainty": {"en": "Uncertainty", "zh": "\u4e0d\u786e\u5b9a\u5ea6"},
    "diag.bands": {"en": "Bands", "zh": "\u6761\u5e26"},
    "diag.lanes": {"en": "Lanes used/found", "zh": "\u6cf3\u9053 \u5df2\u7528/\u53d1\u73b0"},
    "diag.cursor": {"en": "Cursor (raw)", "zh": "\u5149\u6807\uff08\u539f\u59cb\uff09"},
    "diag.reason": {"en": "Reason:", "zh": "\u539f\u56e0\uff1a"},
    "diag.warnings": {"en": "Warnings:", "zh": "\u8b66\u544a\uff1a"},
    "diag.none": {"en": "none (no rotation)", "zh": "\u65e0\uff08\u672a\u65cb\u8f6c\uff09"},
    "diag.not_run": {"en": "not run", "zh": "\u672a\u8fd0\u884c"},
    "diag.dash": {"en": "-", "zh": "-"},
    "diag.no_warnings": {"en": "(none)", "zh": "\uff08\u65e0\uff09"},
    "diag.no_detections": {"en": "no detections", "zh": "\u65e0\u68c0\u6d4b"},
    "diag.failed_load": {"en": "failed to load", "zh": "\u52a0\u8f7d\u5931\u8d25"},
    "diag.accepted": {"en": "accepted", "zh": "\u5df2\u63a5\u53d7"},
    "diag.refused": {"en": "refused", "zh": "\u5df2\u62d2\u7edd"},
    "diag.degree": {"en": "deg", "zh": "\u5ea6"},
    "source.auto": {"en": "auto", "zh": "\u81ea\u52a8"},
    "source.manual": {"en": "manual", "zh": "\u624b\u52a8"},

    # -- status line / busy label -----------------------------------------
    "status.initial": {"en": "Load an image, or a folder of gels, to begin.",
                       "zh": "\u8bf7\u52a0\u8f7d\u4e00\u5f20\u56fe\u50cf\uff0c\u6216\u4e00\u4e2a\u51dd\u80f6\u56fe\u50cf\u6587\u4ef6\u5939\u4ee5\u5f00\u59cb\u3002"},
    "status.idle": {"en": "Idle", "zh": "\u7a7a\u95f2"},
    "status.estimating": {"en": "Estimating...", "zh": "\u4f30\u8ba1\u4e2d\u2026"},
    "status.no_folder": {"en": "no folder selected", "zh": "\u672a\u9009\u62e9\u6587\u4ef6\u5939"},
    "status.folder_count": {"en": "folder: {name}    |    {count} image(s)",
                            "zh": "\u6587\u4ef6\u5939\uff1a{name}    |    {count} \u5f20\u56fe\u50cf"},
    "status.samples_in": {"en": "{count} sample(s) in {name}",
                          "zh": "{name} \u4e2d\u6709 {count} \u4e2a\u6837\u672c"},
    "status.estimating_named": {"en": "estimating {name} ...",
                                "zh": "\u6b63\u5728\u4f30\u8ba1 {name} \u2026"},
    "status.estimate_failed": {"en": "estimate failed", "zh": "\u4f30\u8ba1\u5931\u8d25"},
    "status.applied": {"en": "applied {angle} deg", "zh": "\u5df2\u5e94\u7528 {angle} \u5ea6"},
    "status.estimate_refused": {"en": "estimate refused", "zh": "\u4f30\u8ba1\u88ab\u62d2\u7edd"},
    "status.auto_crop_size": {"en": "auto crop: {w} x {h} px",
                              "zh": "\u81ea\u52a8\u88c1\u526a\uff1a{w} x {h} \u50cf\u7d20"},
    "status.crop_cleared": {"en": "crop cleared", "zh": "\u5df2\u6e05\u9664\u88c1\u526a"},
    "status.composite_off_few_lanes": {
        "en": "Composite crop off \u2014 fewer than two lanes in the crop.",
        "zh": "\u5df2\u5173\u95ed\u590d\u5408\u88c1\u526a \u2014\u2014 \u88c1\u526a\u533a\u57df\u5185\u7684\u6cf3\u9053\u5c11\u4e8e\u4e24\u6761\u3002",
    },
    "status.auto_crop_first": {
        "en": "Auto crop first \u2014 composite cropping joins the lanes it finds.",
        "zh": "\u8bf7\u5148\u81ea\u52a8\u88c1\u526a \u2014\u2014 \u590d\u5408\u88c1\u526a\u4f1a\u5408\u5e76\u5176\u627e\u5230\u7684\u6cf3\u9053\u3002",
    },
    "status.lanes_reidentified": {
        "en": "Lanes re-identified from the crop \u2014 {count} lane(s).",
        "zh": "\u5df2\u4ece\u88c1\u526a\u533a\u57df\u91cd\u65b0\u8bc6\u522b\u6cf3\u9053 \u2014\u2014 {count} \u6761\u3002",
    },
    "status.composite_off_crop_applies": {
        "en": "Composite crop off \u2014 the crop box applies again.",
        "zh": "\u5df2\u5173\u95ed\u590d\u5408\u88c1\u526a \u2014\u2014 \u91cd\u65b0\u5e94\u7528\u88c1\u526a\u6846\u3002",
    },
    "status.composite_spec": {
        "en": "Composite crop: {order} \u2014 {count} lane(s) joined in that order.",
        "zh": "\u590d\u5408\u88c1\u526a\uff1a{order} \u2014\u2014 \u6309\u6b64\u987a\u5e8f\u5408\u5e76 {count} \u6761\u6cf3\u9053\u3002",
    },
    "status.lanes_already_even": {"en": "Lanes already even.", "zh": "\u6cf3\u9053\u5df2\u7b49\u5bbd\u3002"},
    "status.lanes_evened": {"en": "Lanes evened.", "zh": "\u5df2\u7b49\u5bbd\u6cf3\u9053\u3002"},
    "status.lanes_nudged": {"en": "Lanes nudged off the bands.",
                            "zh": "\u5df2\u4f7f\u6cf3\u9053\u907f\u5f00\u6761\u5e26\u3002"},
    "status.lanes_evened_nudged": {"en": "Lanes evened and nudged off the bands.",
                                   "zh": "\u5df2\u7b49\u5bbd\u5e76\u4f7f\u6cf3\u9053\u907f\u5f00\u6761\u5e26\u3002"},
    "status.boundaries_still_cut": {"en": "{count} boundary(ies) still cut a band.",
                                    "zh": "\u4ecd\u6709 {count} \u6761\u8fb9\u754c\u5207\u5272\u6761\u5e26\u3002"},
    "status.lane_verb_removed": {"en": "removed", "zh": "\u5df2\u79fb\u9664"},
    "status.lane_verb_added": {"en": "added", "zh": "\u5df2\u6dfb\u52a0"},
    "status.lane_click": {"en": "Lane {lane} {verb} \u2014 the figure is lanes {shown}.",
                          "zh": "\u6cf3\u9053 {lane} {verb} \u2014\u2014 \u5f53\u524d\u4e3a\u6cf3\u9053 {shown}\u3002"},
    "status.no_border_added": {
        "en": "No border added: too close to the border beside it, or outside the lanes.",
        "zh": "\u672a\u6dfb\u52a0\u8fb9\u754c\uff1a\u8ddd\u79bb\u76f8\u90bb\u8fb9\u754c\u592a\u8fd1\uff0c\u6216\u4f4d\u4e8e\u6cf3\u9053\u533a\u57df\u4e4b\u5916\u3002",
    },
    "status.border_added": {"en": "Border added \u2014 {count} lanes now.",
                            "zh": "\u5df2\u6dfb\u52a0\u8fb9\u754c \u2014\u2014 \u73b0\u6709 {count} \u6761\u6cf3\u9053\u3002"},
    "status.border_stays": {
        "en": "That border stays: an outer boundary can be dropped right down to a "
              "single lane, and an inner join keeps at least two lanes.",
        "zh": "该边界保留：外侧边界可删除至仅剩一条泳道；内侧合并至少需要两条泳道。",
    },
    "status.border_removed": {"en": "Border removed \u2014 {count} lanes now.",
                              "zh": "\u5df2\u79fb\u9664\u8fb9\u754c \u2014\u2014 \u73b0\u6709 {count} \u6761\u6cf3\u9053\u3002"},
    "status.saved": {"en": "saved {name}", "zh": "\u5df2\u4fdd\u5b58 {name}"},
    "status.copied": {"en": "copied TIFF to clipboard",
                      "zh": "\u5df2\u5c06 TIFF \u590d\u5236\u5230\u526a\u8d34\u677f"},
    "status.none_placeholder": {"en": "(none)", "zh": "\uff08\u65e0\uff09"},

    # -- dialog titles + messages -----------------------------------------
    "dlg.open_folder": {"en": "Open folder", "zh": "\u6253\u5f00\u6587\u4ef6\u5939"},
    "dlg.load_error": {"en": "Load error", "zh": "\u52a0\u8f7d\u9519\u8bef"},
    "dlg.auto_rotate": {"en": "Auto-rotate", "zh": "\u81ea\u52a8\u65cb\u8f6c"},
    "dlg.estimate_failed": {"en": "Estimate failed", "zh": "\u4f30\u8ba1\u5931\u8d25"},
    "dlg.auto_crop": {"en": "Auto crop", "zh": "\u81ea\u52a8\u88c1\u526a"},
    "dlg.save_as": {"en": "Save as", "zh": "\u53e6\u5b58\u4e3a"},
    "dlg.copy_clipboard": {"en": "Copy to clipboard", "zh": "\u590d\u5236\u5230\u526a\u8d34\u677f"},
    "msg.open_image_first": {"en": "Open an image first.", "zh": "\u8bf7\u5148\u6253\u5f00\u4e00\u5f20\u56fe\u50cf\u3002"},
    "msg.not_a_folder": {"en": "Not a folder:\n{path}",
                         "zh": "\u4e0d\u662f\u6587\u4ef6\u5939\uff1a\n{path}"},
    "msg.no_images": {"en": "No supported images found in:\n{path}\n\nSupported: {exts}",
                      "zh": "\u5728\u4ee5\u4e0b\u4f4d\u7f6e\u672a\u627e\u5230\u53d7\u652f\u6301\u7684\u56fe\u50cf\uff1a\n{path}\n\n"
                            "\u652f\u6301\uff1a{exts}"},
    "msg.could_not_load": {"en": "could not load {name}:\n{exc}",
                           "zh": "\u65e0\u6cd5\u52a0\u8f7d {name}\uff1a\n{exc}"},
    "msg.band_detection_failed": {"en": "band detection failed for {name}:\n{exc}",
                                  "zh": "{name} \u7684\u6761\u5e26\u68c0\u6d4b\u5931\u8d25\uff1a\n{exc}"},
    "msg.no_bands": {"en": "No bands detected in {name}.",
                     "zh": "\u5728 {name} \u4e2d\u672a\u68c0\u6d4b\u5230\u6761\u5e26\u3002"},
    "msg.run_auto_rotate_first": {"en": "Run Auto rotate before cropping.",
                                  "zh": "\u88c1\u526a\u524d\u8bf7\u5148\u6267\u884c\u81ea\u52a8\u65cb\u8f6c\u3002"},
    "msg.composite_rotation_only": {
        "en": "Composite crop is set, but Save/copy is \u201cRotation only\u201d. "
              "Choose \u201cRotation + crop\u201d, or untick composite cropping.",
        "zh": "\u5df2\u8bbe\u7f6e\u590d\u5408\u88c1\u526a\uff0c\u4f46\u4fdd\u5b58/\u590d\u5236\u4e3a\u201c\u4ec5\u65cb\u8f6c\u201d\u3002"
              "\u8bf7\u9009\u62e9\u201c\u65cb\u8f6c + \u88c1\u526a\u201d\uff0c\u6216\u53d6\u6d88\u52fe\u9009\u590d\u5408\u88c1\u526a\u3002",
    },
    "msg.no_image_open": {"en": "No image is open.", "zh": "\u672a\u6253\u5f00\u56fe\u50cf\u3002"},
    "msg.could_not_save": {"en": "could not save:\n{exc}",
                           "zh": "\u65e0\u6cd5\u4fdd\u5b58\uff1a\n{exc}"},
    "msg.could_not_copy": {"en": "could not copy:\n{exc}",
                           "zh": "\u65e0\u6cd5\u590d\u5236\uff1a\n{exc}"},

    # -- file dialogs ------------------------------------------------------
    "fd.select_image": {"en": "Select an image", "zh": "\u9009\u62e9\u56fe\u50cf"},
    "fd.select_image_folder": {"en": "Select image folder", "zh": "\u9009\u62e9\u56fe\u50cf\u6587\u4ef6\u5939"},
    "fd.save_corrected": {"en": "Save corrected image", "zh": "\u4fdd\u5b58\u6821\u6b63\u540e\u7684\u56fe\u50cf"},

    # -- language switch ---------------------------------------------------

    "lang.restart_title": {"en": "Language", "zh": "\u8bed\u8a00"},
    "lang.restart_message": {"en": "The language will change after you restart the app.",
                             "zh": "\u8bed\u8a00\u5c06\u5728\u91cd\u542f\u5e94\u7528\u540e\u5207\u6362\u3002"},

    # -- widgets -----------------------------------------------------------
    "nav.no_image": {"en": "no image", "zh": "\u65e0\u56fe\u50cf"},
    "viewer.empty": {"en": "Open an image folder to start\n(ctrl+O)",
                     "zh": "\u6253\u5f00\u4e00\u4e2a\u56fe\u50cf\u6587\u4ef6\u5939\u4ee5\u5f00\u59cb\n(ctrl+O)"},

    # -- about -------------------------------------------------------------
    "about.title": {"en": "About geltilt", "zh": "\u5173\u4e8e geltilt"},
    "about.header": {"en": "geltilt \u2014 gel electrophoresis image editor",
                     "zh": "geltilt \u2014\u2014 \u51dd\u80f6\u7535\u6cf3\u56fe\u50cf\u7f16\u8f91\u5668"},
    "about.license_missing": {"en": "(license text not found)",
                              "zh": "\uff08\u672a\u627e\u5230\u8bb8\u53ef\u8bc1\u6587\u672c\uff09"},

    # -- lane-spec messages (SPEC-COMPOSITE) -------------------------------
    "lane_spec.empty": {"en": "Enter a lane spec, e.g. 3,1-2,4-6.",
                        "zh": "\u8bf7\u8f93\u5165\u6cf3\u9053\u89c4\u683c\uff0c\u4f8b\u5982 3,1-2,4-6\u3002"},
    "lane_spec.braces": {"en": "The fan-out form {{..}} is not supported yet.",
                         "zh": "\u5c1a\u4e0d\u652f\u6301\u6269\u5c55\u5f62\u5f0f {{..}}\u3002"},
    "lane_spec.empty_term": {"en": "There is an empty term (a stray comma).",
                             "zh": "\u5b58\u5728\u7a7a\u9879\uff08\u591a\u4f59\u7684\u9017\u53f7\uff09\u3002"},
    "lane_spec.unparsed": {"en": "Could not parse \u201c{term}\u201d.",
                           "zh": "\u65e0\u6cd5\u89e3\u6790\u201c{term}\u201d\u3002"},
    "lane_spec.not_positive": {"en": "Lane numbers start at 1.",
                               "zh": "\u6cf3\u9053\u7f16\u53f7\u4ece 1 \u5f00\u59cb\u3002"},
    "lane_spec.out_of_range": {"en": "Lane {term} is out of range \u2014 {count} lane(s).",
                               "zh": "\u6cf3\u9053 {term} \u8d85\u51fa\u8303\u56f4 \u2014\u2014 \u5171 {count} \u6761\u6cf3\u9053\u3002"},
    "lane_spec.reversed": {"en": "Reversed range \u201c{term}\u201d.",
                           "zh": "\u53cd\u5411\u533a\u95f4\u201c{term}\u201d\u3002"},
}


def get_language() -> str:
    """The active language code."""
    return _LANG


def set_language(code: Optional[str]) -> str:
    """Set the active language (unknown codes fall back to the default)."""
    global _LANG
    _LANG = code if code in LANGUAGES else DEFAULT_LANGUAGE
    return _LANG


def available_languages() -> List[str]:
    """Language codes in display order."""
    return list(LANGUAGES)


def language_label(code: str) -> str:
    """The native display name for a language code."""
    return LANGUAGES.get(code, code)


def language_code(label: str) -> Optional[str]:
    """Reverse of :func:`language_label` (``None`` if unknown)."""
    for code, name in LANGUAGES.items():
        if name == label:
            return code
    return None


def tr(key: str, language: Optional[str] = None, **kwargs: object) -> str:
    """Look up ``key`` in the active (or given) language.

    Unknown keys return the key itself so a missing translation degrades safely
    rather than raising. ``str.format`` placeholders are substituted from
    ``kwargs`` (and doubled braces are collapsed); a formatting error falls back
    to the raw template.
    """
    lang = language if language in LANGUAGES else _LANG
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(lang) or entry.get(DEFAULT_LANGUAGE) or key
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return text


# --------------------------------------------------------------------------- #
# Algorithm diagnostics (emitted by ``geltilt.ladders`` as English templates)  #
# --------------------------------------------------------------------------- #
# The rotation core is shared with the CLI and formats numbers into its reason
# and warning strings, so it is left untouched.  Here we pattern-match the known
# templates and render them in Chinese; anything unrecognised stays as-is.
_REASON_PATTERNS: List[Tuple[re.Pattern, Callable[[re.Match], str]]] = [
    (re.compile(r"^fewer than (\d+) bands$"),
     lambda m: f"\u6761\u5e26\u5c11\u4e8e {m.group(1)} \u4e2a"),
    (re.compile(r"^no lane-like rung column found$"),
     lambda m: "\u672a\u627e\u5230\u7c7b\u4f3c\u6cf3\u9053\u7684\u6761\u5e26\u5217"),
    (re.compile(r"^no lane with >=(\d+) rungs; used (\d+)-rung column\(s\) \(weak\)$"),
     lambda m: (f"\u6ca1\u6709\u6cf3\u9053\u5177\u6709 >= {m.group(1)} \u4e2a\u6761\u5e26\uff1b"
                f"\u4f7f\u7528\u4e86 {m.group(2)} \u4e2a\u6761\u5e26\u7684\u5217\uff08\u5f31\uff09")),
    (re.compile(r"^lanes disagree by more than ([\d.]+) deg about the median ([\d.]+); "
                r"used the (\d+) agreeing one\(s\)$"),
     lambda m: (f"\u6cf3\u9053\u76f8\u5bf9\u4e2d\u4f4d\u6570 {m.group(2)} \u7684\u5206\u6b67\u8d85\u8fc7 "
                f"{m.group(1)} \u5ea6\uff1b\u91c7\u7528\u4e86 {m.group(3)} \u6761\u4e00\u81f4\u7684\u6cf3\u9053")),
    (re.compile(r"^implausible tilt ([+\-][\d.]+) deg exceeds \+\-([\d.]+) deg; "
                r"grouping suspect(.*)$"),
     lambda m: (f"\u503e\u659c\u89d2 {m.group(1)} \u5ea6\u4e0d\u5408\u7406\uff0c\u8d85\u8fc7 "
                f"\u00b1{m.group(2)} \u5ea6\uff1b\u5206\u7ec4\u53ef\u7591"
                + ("\uff083 \u6761\u5e26\u56de\u9000\uff09" if "fallback" in m.group(3) else ""))),
    (re.compile(r"^ok: (\d+) lane\(s\) pooled of (\d+), (\d+) rungs, se ([\d.]+) deg$"),
     lambda m: (f"\u6b63\u5e38\uff1a\u6c47\u603b {m.group(1)} \u6761\u6cf3\u9053\uff08\u5171 {m.group(2)} \u6761\uff09\uff0c"
                f"{m.group(3)} \u4e2a\u6761\u5e26\uff0c\u6807\u51c6\u8bef {m.group(4)} \u5ea6")),
]

_WARNING_PATTERNS: List[Tuple[re.Pattern, Callable[[re.Match], str]]] = [
    (re.compile(r"^(\d+)-rung fallback: no lane with >=(\d+) rungs in this image; a "
                r"(\d+)-rung column is a short lever arm \(angle se ~([\d.]+) deg, "
                r"(\d+) rungs, (\d+) px span\)$"),
     lambda m: (f"{m.group(1)} \u6761\u5e26\u56de\u9000\uff1a\u672c\u56fe\u4e2d\u6ca1\u6709 >= {m.group(2)} "
                f"\u4e2a\u6761\u5e26\u7684\u6cf3\u9053\uff1b{m.group(3)} \u6761\u5e26\u7684\u5217\u6760\u6746\u8fc7\u77ed"
                f"\uff08\u89d2\u5ea6\u6807\u51c6\u8bef ~{m.group(4)} \u5ea6\uff0c{m.group(5)} \u4e2a\u6761\u5e26\uff0c"
                f"{m.group(6)} \u50cf\u7d20\u8de8\u5ea6\uff09")),
    (re.compile(r"^lane (\d+): line R2 ([\d.]+) < ([\d.]+) \((\d+) rungs, ([\d.]+) px of "
                r"lean, angle \+\-([\d.]+) deg\)$"),
     lambda m: (f"\u6cf3\u9053 {m.group(1)}\uff1a\u76f4\u7ebf R2 {m.group(2)} < {m.group(3)}"
                f"\uff08{m.group(4)} \u4e2a\u6761\u5e26\uff0c\u504f\u659c {m.group(5)} \u50cf\u7d20\uff0c"
                f"\u89d2\u5ea6 \u00b1{m.group(6)} \u5ea6\uff09")),
    (re.compile(r"^longest lane spans only (\d+) px \(< (\d+)\): short lever arm$"),
     lambda m: (f"\u6700\u957f\u7684\u6cf3\u9053\u4ec5\u8de8\u8d8a {m.group(1)} \u50cf\u7d20"
                f"\uff08< {m.group(2)}\uff09\uff1a\u6760\u6746\u8fc7\u77ed")),
    (re.compile(r"^lanes disagree ([\d.]+)x more than their own errors predict "
                r"\(Q/dof ([\d.]+)\): the pooled se already carries this inflation$"),
     lambda m: (f"\u6cf3\u9053\u95f4\u7684\u5206\u6b67\u6bd4\u5176\u81ea\u8eab\u8bef\u5dee\u9884\u6d4b\u9ad8 "
                f"{m.group(1)} \u500d\uff08Q/dof {m.group(2)}\uff09\uff1a\u6c47\u603b\u6807\u51c6\u8bef\u5df2\u5305\u542b"
                f"\u8be5\u81a8\u80c0")),
    (re.compile(r"^single lane: no independent consensus, the answer is that lane alone "
                r"\(se ([\d.]+) deg\)$"),
     lambda m: (f"\u5355\u6cf3\u9053\uff1a\u65e0\u72ec\u7acb\u5171\u8bc6\uff0c\u7ed3\u679c\u4ec5\u6765\u81ea\u8be5\u6cf3\u9053"
                f"\uff08\u6807\u51c6\u8bef {m.group(1)} \u5ea6\uff09")),
    (re.compile(r"^empirical over-rotation correction applied: ([+\-][\d.]+) -> "
                r"([+\-][\d.]+) deg$"),
     lambda m: (f"\u5df2\u5e94\u7528\u7ecf\u9a8c\u6027\u8fc7\u65cb\u8f6c\u6821\u6b63\uff1a{m.group(1)} -> "
                f"{m.group(2)} \u5ea6")),
]


def localize_diagnostic(text: str) -> str:
    """Render an algorithm reason/warning string in the active language.

    English (the default) and unrecognised strings are returned unchanged.
    """
    if not text or _LANG == "en":
        return text
    for patterns in (_REASON_PATTERNS, _WARNING_PATTERNS):
        for pattern, render in patterns:
            match = pattern.match(text)
            if match:
                return render(match)
    return text
"""Tests for the template contract — slot vocabulary, required IDs, build markers.

The contract module is the public surface that defines what a Chatnificent
template MUST provide so the framework's CSS and JavaScript work. Every
shipped template is validated against it; devs forking templates can run
the same validator against their own folders.
"""

from pathlib import Path

import chatnificent as chat
import pytest

# ---------------------------------------------------------------------------
# Constants — the locked public vocabulary
# ---------------------------------------------------------------------------


class TestBodySlots:
    def test_body_slots_is_a_tuple(self):
        # Tuples enforce immutability — slot names are public API after release.
        assert isinstance(chat.templates._contract.BODY_SLOTS, tuple)

    def test_body_slots_contains_all_16_locked_names(self):
        expected = {
            "brand",
            "slogan",
            "header-begin",
            "header-end",
            "sidebar-begin",
            "sidebar-end",
            "welcome-message",
            "messages-begin",
            "messages-end",
            "composer-leading",
            "composer-primary",
            "composer-trailing",
            "composer-send",
            "composer-attachments",
            "footer-begin",
            "footer-end",
        }
        assert set(chat.templates._contract.BODY_SLOTS) == expected

    def test_body_slots_has_no_duplicates(self):
        assert len(chat.templates._contract.BODY_SLOTS) == len(
            set(chat.templates._contract.BODY_SLOTS)
        )


class TestRequiredElementIds:
    def test_required_ids_is_a_tuple(self):
        assert isinstance(chat.templates._contract.REQUIRED_ELEMENT_IDS, tuple)

    def test_required_ids_contains_all_11_anchors(self):
        expected = {
            "messages",
            "input",
            "send",
            "sidebar",
            "convo-list",
            "chat-wrap",
            "welcome",
            "welcome-message",
            "sidebar-toggle",
            "theme-toggle",
            "new-chat-btn",
        }
        assert set(chat.templates._contract.REQUIRED_ELEMENT_IDS) == expected


class TestBuildMarkers:
    def test_build_markers_is_a_tuple(self):
        assert isinstance(chat.templates._contract.BUILD_MARKERS, tuple)

    def test_build_markers_contains_three_entries(self):
        assert set(chat.templates._contract.BUILD_MARKERS) == {
            "VENDOR",
            "STYLES",
            "SCRIPTS",
        }


# ---------------------------------------------------------------------------
# chat.templates._contract.validate_template — runs against a template folder
# ---------------------------------------------------------------------------


def _make_valid_template(root: Path) -> Path:
    """Write a minimal but contract-compliant template folder under ``root``."""
    tpl = root / "tpl"
    tpl.mkdir()
    slot_divs = "\n".join(
        f'<div data-slot="{name}"></div>'
        for name in chat.templates._contract.BODY_SLOTS
    )
    id_divs = "\n".join(
        f'<div id="{name}"></div>'
        for name in chat.templates._contract.REQUIRED_ELEMENT_IDS
    )
    (tpl / "template.html").write_text(
        f"""<!doctype html>
<html>
<head>
  <title>App</title>
  <!-- VENDOR -->
  <style><!-- STYLES --></style>
  <script><!-- SCRIPTS --></script>
</head>
<body>
{slot_divs}
{id_divs}
</body>
</html>
""",
        encoding="utf-8",
    )
    (tpl / "styles.css").write_text("/* css */", encoding="utf-8")
    (tpl / "scripts.js").write_text("/* js */", encoding="utf-8")
    (tpl / "vendor").mkdir()
    return tpl


class TestValidateTemplate:
    def test_passes_on_a_compliant_template_folder(self, tmp_path):
        tpl = _make_valid_template(tmp_path)
        # Should not raise.
        chat.templates._contract.validate_template(tpl)

    def test_accepts_string_name_resolved_against_builtin_dir(self):
        # The shipped 'default' template MUST validate.
        chat.templates._contract.validate_template("default")

    def test_returns_none_on_success(self, tmp_path):
        tpl = _make_valid_template(tmp_path)
        assert chat.templates._contract.validate_template(tpl) is None

    def test_raises_on_missing_template_html(self, tmp_path):
        tpl = tmp_path / "tpl"
        tpl.mkdir()
        with pytest.raises(
            chat.templates._contract.ValidationError, match="template.html"
        ):
            chat.templates._contract.validate_template(tpl)

    def test_raises_when_a_body_slot_div_is_missing(self, tmp_path):
        tpl = _make_valid_template(tmp_path)
        html = (tpl / "template.html").read_text(encoding="utf-8")
        html = html.replace('<div data-slot="brand"></div>', "")
        (tpl / "template.html").write_text(html, encoding="utf-8")
        with pytest.raises(chat.templates._contract.ValidationError, match="brand"):
            chat.templates._contract.validate_template(tpl)

    def test_raises_when_a_required_id_is_missing(self, tmp_path):
        tpl = _make_valid_template(tmp_path)
        html = (tpl / "template.html").read_text(encoding="utf-8")
        html = html.replace('<div id="messages"></div>', "")
        (tpl / "template.html").write_text(html, encoding="utf-8")
        with pytest.raises(chat.templates._contract.ValidationError, match="messages"):
            chat.templates._contract.validate_template(tpl)

    def test_raises_when_a_build_marker_is_missing(self, tmp_path):
        tpl = _make_valid_template(tmp_path)
        html = (tpl / "template.html").read_text(encoding="utf-8")
        html = html.replace("<!-- VENDOR -->", "")
        (tpl / "template.html").write_text(html, encoding="utf-8")
        with pytest.raises(chat.templates._contract.ValidationError, match="VENDOR"):
            chat.templates._contract.validate_template(tpl)

    def test_raises_when_title_is_missing_from_head(self, tmp_path):
        tpl = _make_valid_template(tmp_path)
        html = (tpl / "template.html").read_text(encoding="utf-8")
        html = html.replace("<title>App</title>", "")
        (tpl / "template.html").write_text(html, encoding="utf-8")
        with pytest.raises(chat.templates._contract.ValidationError, match=r"<title>"):
            chat.templates._contract.validate_template(tpl)

    def test_raises_on_unknown_builtin_template_name(self):
        with pytest.raises(
            chat.templates._contract.ValidationError, match="not-a-real-template"
        ):
            chat.templates._contract.validate_template("not-a-real-template")

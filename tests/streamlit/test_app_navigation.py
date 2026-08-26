from pathlib import Path

from streamlit.testing.v1 import AppTest

import aplit

APP_PATH = Path(aplit.__file__).parent / "app.py"


def test_navigation_exposes_results_submit_and_runs(monkeypatch) -> None:
    monkeypatch.delenv("APLIT_SITE_CONFIG", raising=False)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert app.radio[0].label == "Workspace"
    assert app.radio[0].options == ["Results", "Submit", "Runs"]

    app.radio[0].set_value("Submit").run()
    assert any(title.value == "Submit run" for title in app.title)
    assert any("Submission is not configured" in item.value for item in app.info)

    app.radio[0].set_value("Runs").run()
    assert any(title.value == "Runs" for title in app.title)


def test_configured_submit_form_switches_exclusive_modes(
    monkeypatch, fake_site_file: Path
) -> None:
    monkeypatch.setenv("APLIT_SITE_CONFIG", str(fake_site_file))
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.radio[0].set_value("Submit").run()

    assert not app.exception
    assert any(item.label == "Project name" for item in app.text_input)
    mode = next(item for item in app.radio if item.label == "Input mode")
    assert mode.value == "Direct complexes"
    assert any(item.label == "Complexes" for item in app.text_area)
    assert not any(item.label == "Baits" for item in app.text_area)

    mode.set_value("Bait–prey screen").run()
    assert any(item.label == "Baits" for item in app.text_area)
    assert any(item.label == "Preys" for item in app.text_area)
    assert not any(item.label == "Complexes" for item in app.text_area)


def test_submit_validation_and_launch_flow(
    monkeypatch, fake_site_file: Path, fake_site
) -> None:
    fake_site.snakemake_executable.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in *--dry-run*) echo 'Job stats:'; echo 'all 1'; echo 'total 1';; esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("APLIT_SITE_CONFIG", str(fake_site_file))
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.radio[0].set_value("Submit").run()

    next(item for item in app.text_input if item.label == "Project name").set_value(
        "gui-run"
    )
    next(item for item in app.text_area if item.label == "Complexes").set_value("P1+P2")
    next(item for item in app.button if item.label == "Validate").click().run()

    assert not app.exception
    assert any("Validation succeeded" in item.value for item in app.success)
    assert any(item.label == "Submit run" for item in app.button)

    next(item for item in app.button if item.label == "Submit run").click().run()
    assert not app.exception
    assert app.radio[0].value == "Runs"
    assert any(title.value == "Runs" for title in app.title)


def test_submit_reports_non_mapping_overrides(
    monkeypatch, fake_site_file: Path
) -> None:
    monkeypatch.setenv("APLIT_SITE_CONFIG", str(fake_site_file))
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.radio[0].set_value("Submit").run()

    next(item for item in app.text_input if item.label == "Project name").set_value(
        "bad-overrides"
    )
    next(item for item in app.text_area if item.label == "Complexes").set_value("P1+P2")
    next(
        item
        for item in app.text_area
        if item.label == "Advanced overrides (allowlisted YAML)"
    ).set_value("- not\n- a\n- mapping")
    next(item for item in app.button if item.label == "Validate").click().run()

    assert not app.exception
    assert any("must be a YAML mapping" in item.value for item in app.error)

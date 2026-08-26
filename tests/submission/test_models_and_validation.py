import pytest

from aplit.submission.models import InputMode, RunSpec, SiteConfig, UploadedFasta
from aplit.submission.validator import (
    parse_allowlisted_overrides,
    preflight_site,
    validate_run_spec,
)


def test_input_modes_are_mutually_exclusive(fake_site: SiteConfig) -> None:
    spec = RunSpec(
        project_name="mixed",
        input_mode=InputMode.DIRECT,
        backend="af3",
        direct_complexes=("P01258+P01579",),
        baits=("P1",),
        preys=("P2",),
    )

    result = validate_run_spec(spec, fake_site)

    assert not result.valid
    assert any("cannot include bait" in error for error in result.errors)


def test_large_bait_prey_screen_warns(fake_site: SiteConfig) -> None:
    spec = RunSpec(
        project_name="screen",
        input_mode=InputMode.BAIT_PREY,
        backend="af3",
        baits=tuple(f"B{i}" for i in range(40)),
        preys=tuple(f"P{i}" for i in range(30)),
    )

    result = validate_run_spec(spec, fake_site)

    assert result.valid
    assert "1,200" in result.warnings[0]


def test_override_allowlist_rejects_paths() -> None:
    with pytest.raises(ValueError, match="Unsupported advanced override"):
        parse_allowlisted_overrides({"backend_weights_directory": "/tmp/private"})


def test_upload_content_affects_fingerprint() -> None:
    base = {
        "project_name": "upload",
        "input_mode": InputMode.DIRECT,
        "backend": "af3",
        "direct_complexes": ("protein",),
    }
    first = RunSpec(**base, uploads=(UploadedFasta("protein.fasta", b">p\nAAAA\n"),))
    second = RunSpec(**base, uploads=(UploadedFasta("protein.fasta", b">p\nAAAT\n"),))

    assert first.fingerprint() != second.fingerprint()


def test_site_preflight_accepts_complete_local_fixture(fake_site: SiteConfig) -> None:
    result = preflight_site(fake_site)

    assert result.valid, result.errors


def test_fold_shell_metacharacters_are_rejected(fake_site: SiteConfig) -> None:
    result = validate_run_spec(
        RunSpec(
            project_name="unsafe",
            input_mode=InputMode.DIRECT,
            backend="af3",
            direct_complexes=("P1;touch_bad+P2",),
        ),
        fake_site,
    )

    assert not result.valid
    assert any("unsafe characters" in error for error in result.errors)


def test_server_fasta_content_is_validated(fake_site: SiteConfig, tmp_path) -> None:
    invalid_fasta = tmp_path / "invalid.fasta"
    invalid_fasta.write_text("not-a-fasta\n", encoding="utf-8")

    result = validate_run_spec(
        RunSpec(
            project_name="invalid-server-fasta",
            input_mode=InputMode.DIRECT,
            backend="af3",
            direct_complexes=(str(invalid_fasta),),
        ),
        fake_site,
    )

    assert not result.valid
    assert any("before its first header" in error for error in result.errors)

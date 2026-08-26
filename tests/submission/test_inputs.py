from pathlib import Path

import pytest

from aplit.submission.errors import ValidationError
from aplit.submission.inputs import (
    parse_fasta,
    rewrite_upload_aliases,
    stage_uploads,
    upload_aliases,
)
from aplit.submission.models import UploadedFasta


def test_parse_multirecord_fasta() -> None:
    records = parse_fasta(b">one description\nAAAA\n>two\nCCDD\n", source="x.fa")

    assert [record.identifier for record in records] == ["one", "two"]
    assert records[0].sequence == "AAAA"


def test_parse_fasta_rejects_sequence_before_header() -> None:
    with pytest.raises(ValidationError, match="before its first header"):
        parse_fasta(b"AAAA\n", source="x.fa")


def test_stage_upload_and_rewrite_alias(tmp_path: Path) -> None:
    staged = stage_uploads([UploadedFasta("My protein.fa", b">p\nAAAA\n")], tmp_path)

    assert staged[0].alias == "My_protein"
    assert staged[0].path.read_text().startswith(">p\nAAAA")
    assert rewrite_upload_aliases("My_protein:2+P01258", staged) == (
        f"{staged[0].path}:2+P01258"
    )


def test_multirecord_upload_aliases_use_record_ids() -> None:
    aliases = upload_aliases(
        [UploadedFasta("many.fasta", b">alpha one\nAAAA\n>beta two\nCCCC\n")]
    )

    assert aliases == ("alpha", "beta")

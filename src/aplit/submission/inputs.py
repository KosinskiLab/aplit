"""FASTA validation, staging, and safe fold-specification rewriting."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .errors import ValidationError
from .models import UploadedFasta
from .validator import is_within

SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")
SEQUENCE_RE = re.compile(r"^[A-Za-z*.-]+$")


@dataclass(frozen=True)
class FastaRecord:
    identifier: str
    description: str
    sequence: str


@dataclass(frozen=True)
class StagedUpload:
    original_name: str
    alias: str
    path: Path
    sha256: str
    records: tuple[FastaRecord, ...]


def safe_stem(value: str, fallback: str = "sequence") -> str:
    stem = Path(value).stem.strip()
    sanitized = SAFE_STEM_RE.sub("_", stem).strip("._-")
    return sanitized[:80] or fallback


def parse_fasta(content: bytes, *, source: str) -> tuple[FastaRecord, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"FASTA {source} is not UTF-8 text") from exc

    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def finish_record() -> None:
        if header is None:
            return
        sequence = "".join(sequence_parts).replace(" ", "").upper()
        if not sequence:
            raise ValidationError(f"FASTA record '{header}' in {source} is empty")
        if not SEQUENCE_RE.fullmatch(sequence):
            raise ValidationError(
                f"FASTA record '{header}' in {source} contains invalid sequence characters"
            )
        identifier = header.split(maxsplit=1)[0]
        if not identifier:
            raise ValidationError(f"FASTA {source} contains an empty header")
        records.append(FastaRecord(identifier, header, sequence))

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish_record()
            header = line[1:].strip()
            sequence_parts = []
            if not header:
                raise ValidationError(
                    f"FASTA {source} has an empty header on line {line_number}"
                )
        else:
            if header is None:
                raise ValidationError(
                    f"FASTA {source} has sequence data before its first header"
                )
            sequence_parts.append(line)
    finish_record()
    if not records:
        raise ValidationError(f"FASTA {source} contains no records")
    return tuple(records)


def _format_record(record: FastaRecord) -> bytes:
    chunks = [
        record.sequence[index : index + 80]
        for index in range(0, len(record.sequence), 80)
    ]
    return (f">{record.description}\n" + "\n".join(chunks) + "\n").encode("utf-8")


def upload_aliases(uploads: Iterable[UploadedFasta]) -> tuple[str, ...]:
    """Return the exact aliases that staging will assign, without writing files."""

    result: list[str] = []
    used: set[str] = set()
    for index, upload in enumerate(uploads, start=1):
        records = parse_fasta(upload.content, source=upload.name)
        base = safe_stem(upload.name, f"upload_{index}")
        for record in records:
            alias_base = (
                base if len(records) == 1 else safe_stem(record.identifier, base)
            )
            alias = alias_base
            suffix = 2
            while alias.casefold() in used:
                alias = f"{alias_base}_{suffix}"
                suffix += 1
            used.add(alias.casefold())
            result.append(alias)
    return tuple(result)


def stage_uploads(
    uploads: Iterable[UploadedFasta], destination: Path
) -> tuple[StagedUpload, ...]:
    uploads = tuple(uploads)
    destination.mkdir(parents=True, exist_ok=True)
    staged: list[StagedUpload] = []
    assigned_aliases = iter(upload_aliases(uploads))
    for upload in uploads:
        records = parse_fasta(upload.content, source=upload.name)
        for record in records:
            alias = next(assigned_aliases)
            path = destination / f"{alias}.fasta"
            path.write_bytes(_format_record(record))
            staged.append(
                StagedUpload(
                    original_name=upload.name,
                    alias=alias,
                    path=path,
                    sha256=upload.sha256,
                    records=(record,),
                )
            )
    return tuple(staged)


def validate_server_fasta(path: Path, allowed_roots: tuple[Path, ...]) -> Path:
    if not is_within(path, allowed_roots):
        raise ValidationError(
            f"Server FASTA path is missing or outside allowed roots: {path}"
        )
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValidationError(f"Server FASTA path is not a file: {path}")
    parse_fasta(resolved.read_bytes(), source=str(resolved))
    return resolved


def rewrite_upload_aliases(row: str, uploads: Iterable[StagedUpload]) -> str:
    """Replace an exact fold token matching an upload alias with its staged path."""

    alias_paths = {upload.alias: str(upload.path) for upload in uploads}
    parts: list[str] = []
    for token in row.split("+"):
        base, separator, suffix = token.partition(":")
        replacement = alias_paths.get(base, base)
        parts.append(replacement + (separator + suffix if separator else ""))
    return "+".join(parts)

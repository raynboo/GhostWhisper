from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import soundfile as sf


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}


@dataclass
class ManifestEntry:
    dataset: str
    speaker_id: str
    utterance_id: str
    audio_path: str
    transcript: str
    source_sample_rate: int
    duration_sec: float


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore").strip()


def _audio_info(path: Path) -> tuple[int, float]:
    info = sf.info(str(path))
    return int(info.samplerate), float(info.frames / info.samplerate)


def _iter_audio_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            yield path


def scan_vctk(root: Path) -> list[ManifestEntry]:
    wav_root = root / "wav48_silence_trimmed"
    txt_root = root / "txt"
    if not wav_root.exists():
        wav_root = root / "wav48"

    entries: list[ManifestEntry] = []
    for audio_path in sorted(_iter_audio_files(wav_root if wav_root.exists() else root)):
        speaker_id = audio_path.parent.name
        utterance_id = audio_path.stem
        transcript_path = txt_root / speaker_id / f"{utterance_id}.txt"
        transcript = _safe_read_text(transcript_path) if transcript_path.exists() else ""
        sample_rate, duration_sec = _audio_info(audio_path)
        entries.append(
            ManifestEntry(
                dataset="vctk",
                speaker_id=speaker_id,
                utterance_id=utterance_id,
                audio_path=str(audio_path.resolve()),
                transcript=transcript,
                source_sample_rate=sample_rate,
                duration_sec=duration_sec,
            )
        )
    return entries


def scan_librispeech(root: Path) -> list[ManifestEntry]:
    transcript_cache: dict[Path, dict[str, str]] = {}
    entries: list[ManifestEntry] = []
    for audio_path in sorted(path for path in _iter_audio_files(root) if path.suffix.lower() == ".flac"):
        speaker_id = audio_path.parts[-3]
        chapter_id = audio_path.parts[-2]
        utterance_id = audio_path.stem
        trans_path = audio_path.parent / f"{speaker_id}-{chapter_id}.trans.txt"
        if trans_path not in transcript_cache:
            mapping: dict[str, str] = {}
            for line in _safe_read_text(trans_path).splitlines():
                if not line.strip():
                    continue
                key, text = line.split(" ", 1)
                mapping[key] = text.strip()
            transcript_cache[trans_path] = mapping
        sample_rate, duration_sec = _audio_info(audio_path)
        entries.append(
            ManifestEntry(
                dataset="librispeech",
                speaker_id=speaker_id,
                utterance_id=utterance_id,
                audio_path=str(audio_path.resolve()),
                transcript=transcript_cache[trans_path].get(utterance_id, ""),
                source_sample_rate=sample_rate,
                duration_sec=duration_sec,
            )
        )
    return entries


def scan_common_voice(root: Path) -> list[ManifestEntry]:
    clips_root = root / "clips"
    entries: list[ManifestEntry] = []
    for tsv_name in ["validated.tsv", "train.tsv", "dev.tsv", "test.tsv"]:
        tsv_path = root / tsv_name
        if not tsv_path.exists():
            continue
        with tsv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                relative = row.get("path", "")
                if not relative:
                    continue
                audio_path = clips_root / relative
                if not audio_path.exists():
                    continue
                sample_rate, duration_sec = _audio_info(audio_path)
                entries.append(
                    ManifestEntry(
                        dataset="common_voice",
                        speaker_id=row.get("client_id", "unknown"),
                        utterance_id=audio_path.stem,
                        audio_path=str(audio_path.resolve()),
                        transcript=row.get("sentence", "").strip(),
                        source_sample_rate=sample_rate,
                        duration_sec=duration_sec,
                    )
                )
    return entries


def scan_dataset(root: Path, dataset_type: str) -> list[ManifestEntry]:
    key = dataset_type.lower()
    if key == "vctk":
        return scan_vctk(root)
    if key == "librispeech":
        return scan_librispeech(root)
    if key == "common_voice":
        return scan_common_voice(root)
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def write_manifest(entries: list[ManifestEntry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "speaker_id",
                "utterance_id",
                "audio_path",
                "transcript",
                "source_sample_rate",
                "duration_sec",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.__dict__)

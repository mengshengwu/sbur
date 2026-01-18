import subprocess
import sys
from pathlib import Path


def test_integration_rename(tmp_path):
    d = tmp_path / "media"
    d.mkdir()

    videos = [
        "Nozomanu Fushi [01][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [02][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [03][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [04][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [05][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [06][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [07][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [08][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [09][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [10][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [11][Ma10p_1080p][x265_flac].mkv",
        "Nozomanu Fushi [12][Ma10p_1080p][x265_flac].mkv",
    ]

    subtitles = [
        "Nozomanu Fushi no Boukensha - 01.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 02.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 03.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 04.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 05.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 06.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 07.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 08.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 09.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 10.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 11.Netflix.zh-Hant.srt",
        "Nozomanu Fushi no Boukensha - 12.Netflix.zh-Hant.srt",
    ]

    for name in videos + subtitles:
        (d / name).write_text("")

    result = subprocess.run(
        [sys.executable, "main.py", "--verbose", "-d", str(d)],
        input="y\n",
        text=True,
        cwd=Path(__file__).parent,
    )

    assert result.returncode == 0

    for sub, vid in zip(subtitles, videos):
        old_path = d / sub
        new_path = d / (Path(vid).stem + Path(sub).suffix)
        assert not old_path.exists(), f"Expected {sub} to be renamed"
        assert new_path.exists(), f"Expected renamed file {new_path.name} to exist"

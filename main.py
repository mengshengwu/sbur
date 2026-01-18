import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from nexaai import LLM, LlmChatMessage
from nexaai.nexa_sdk.types import GenerationConfig, SamplerConfig


def scan_videos_subs(directory: Path) -> Tuple[List[Path], List[Path]]:
    videos, subtitles = [], []
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm"}
    SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".sub", ".vtt"}
    for file in directory.iterdir():
        if file.is_file():
            suffix = file.suffix.lower()
            if suffix in VIDEO_EXTENSIONS:
                videos.append(file)
            elif suffix in SUBTITLE_EXTENSIONS:
                subtitles.append(file)

    return sorted(videos), sorted(subtitles)


def _parse_llm_json(text: str) -> Dict[str, str]:
    text = text.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            matches = {}
            for item in parsed:
                if isinstance(item, dict):
                    # Handle {"subtitle": "...", "video": "..."} format
                    if "subtitle" in item and "video" in item:
                        matches[item["subtitle"]] = item["video"]
                    else:
                        matches.update(item)
            return matches
        return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON from response
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    logging.debug("Failed to parse LLM response")
    return {}


def match_subtitles(videos: List[Path], subtitles: List[Path]) -> Dict[str, str]:
    messages: List[LlmChatMessage] = [
        LlmChatMessage(
            role="system",
            content=(
                "You are a deterministic filename matching engine.\n"
                "Your task: match subtitle filenames to video filenames for the same episode.\n\n"
                "Rules:\n"
                "1. Use filename text only.\n"
                "2. Prefer exact episode number matches.\n"
                "3. Episode patterns include: E01, EP01, S01E01, 第01集, [01], - 01.\n"
                "4. One subtitle maps to at most one video.\n"
                "5. Do not guess. Omit if uncertain.\n\n"
                "Output ONLY a JSON object mapping subtitle -> video."
            ),
        ),
        LlmChatMessage(
            role="user",
            content=(
                "Video files:\n"
                f"{json.dumps([p.name for p in videos], ensure_ascii=False)}\n\n"
                "Subtitle files:\n"
                f"{json.dumps([p.name for p in subtitles], ensure_ascii=False)}\n\n"
                "Return ONLY the JSON object."
            ),
        ),
    ]

    llm: LLM = LLM.from_("NexaAI/Qwen3-0.6B-GGUF")
    formatted_prompt = llm.apply_chat_template(
        messages=messages, add_generation_prompt=True
    )

    response = llm.generate(
        formatted_prompt,
        config=GenerationConfig(
            max_tokens=4096,
            sampler_config=SamplerConfig(temperature=0.1, enable_json=True),
        ),
    )
    response_text = response.full_text
    logging.debug(f"LLM response received: {len(response_text)} chars")
    matches = _parse_llm_json(response_text)
    if matches:
        logging.info(f"Matched {len(matches)} file(s)")
    return matches


def display_and_confirm(matches: Dict[str, str]) -> bool:
    if not matches:
        logging.warning("No matches found")
        return False

    while True:
        sys.stdout.flush()
        choice = input("\nConfirm these matches? (y/n): ").lower().strip()
        if choice in ("y", "yes"):
            print()
            return True
        if choice in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'")


def rename_subtitles(matches: Dict[str, str], directory: str) -> int:
    count = 0
    for subtitle_name, video_name in matches.items():
        subtitle_path = Path(directory) / subtitle_name
        new_name = Path(video_name).stem + Path(subtitle_name).suffix
        new_path = Path(directory) / new_name

        try:
            subtitle_path.rename(new_path)
            logging.debug(f"Renamed: {subtitle_name} -> {new_name}")
            count += 1
        except Exception as e:
            logging.error(f"Failed to rename {subtitle_name}: {e}")

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename subtitles to match video files using LLM"
    )
    parser.add_argument(
        "-d",
        "--dir",
        default=".",
        help="Directory containing video and subtitle files (default: current directory)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug mode with verbose output",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="[%(levelname)s] %(message)s", force=True
    )

    dir = Path(args.dir).resolve()
    if not dir.is_dir():
        logging.error(f"'{dir}' is not a valid directory")
        return

    logging.debug(f"Scanning directory: {dir}")
    videos, subtitles = scan_videos_subs(dir)
    if not videos or not subtitles:
        logging.error("No video or subtitle files found in the directory")
        return

    logging.debug(f"Found {len(videos)} video(s) and {len(subtitles)} subtitle(s)")

    matches = match_subtitles(videos, subtitles)

    if display_and_confirm(matches):
        sys.stdout.flush()  # Ensure all output is written before proceeding
        logging.debug("Renaming subtitles...")
        count = rename_subtitles(matches, str(dir))
        logging.debug(f"Successfully renamed {count} subtitle file(s)")
    else:
        logging.debug("Operation cancelled")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from langchain_core.tools import tool
from langchain_core.language_models.llms import LLM as BaseLLM
from langchain_core.outputs import Generation, LLMResult
from nexaai import LLM, LlmChatMessage
from nexaai.nexa_sdk.types import GenerationConfig, SamplerConfig

# Initialize NexaAI LLM
nexaai_llm: LLM = LLM.from_("NexaAI/Qwen3-0.6B-GGUF")

# Video and subtitle file extensions
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".sub", ".vtt"}


class NexaAILangChain(BaseLLM):
    """LangChain wrapper for NexaAI LLM to enable agent usage."""

    def __init__(self, llm: LLM, **kwargs: Any):
        super().__init__(**kwargs)
        self._nexaai_llm = llm

    @property
    def _llm_type(self) -> str:
        return "nexaai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        response = self._nexaai_llm.generate(prompt)
        return response.full_text if hasattr(response, "full_text") else str(response)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for multiple prompts."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)


@tool
def analyze_filenames(videos_json: str, subtitles_json: str) -> str:
    """Analyze video and subtitle filenames to understand their naming patterns.

    This tool helps the LLM understand the naming patterns of files.

    Args:
        videos_json: JSON string of video filenames
        subtitles_json: JSON string of subtitle filenames

    Returns:
        JSON string with file information for analysis
    """
    try:
        videos = json.loads(videos_json)
        subtitles = json.loads(subtitles_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})

    info = {
        "video_count": len(videos),
        "subtitle_count": len(subtitles),
        "videos": videos,
        "subtitles": subtitles,
        "message": "Analyze these filenames and determine which subtitle corresponds to which video based on episode numbers, titles, or patterns in the names.",
    }

    return json.dumps(info, ensure_ascii=False, indent=2)


@tool
def create_mapping(mapping_json: str) -> str:
    """Create a mapping between subtitle and video files.

    Use this tool to create the final mapping after analyzing the filenames.

    Args:
        mapping_json: JSON string with format {"subtitle_filename": "video_filename", ...}

    Returns:
        Confirmation message with the mapping
    """
    try:
        mapping = json.loads(mapping_json)
        return json.dumps(
            {"status": "success", "mapping": mapping, "count": len(mapping)},
            ensure_ascii=False,
            indent=2,
        )
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "Invalid JSON format"})


class SubtitleRenamer:
    """Tool for matching and renaming subtitles using LLM."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def get_files_in_directory(self, directory: str) -> Tuple[List[str], List[str]]:
        """Get all video and subtitle files in the directory.

        Args:
            directory: Path to the directory to scan

        Returns:
            Tuple of (video_files, subtitle_files)
        """
        video_files = []
        subtitle_files = []

        for file in os.listdir(directory):
            file_path = Path(directory) / file
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in VIDEO_EXTENSIONS:
                    video_files.append(file)
                elif suffix in SUBTITLE_EXTENSIONS:
                    subtitle_files.append(file)

        return sorted(video_files), sorted(subtitle_files)

    def match_subtitles_with_llm(
        self, videos: List[str], subtitles: List[str]
    ) -> Dict[str, str]:
        """Use LLM with tools to match subtitle files with video files.

        Args:
            videos: List of video file names
            subtitles: List of subtitle file names

        Returns:
            Dictionary mapping subtitle filename to video filename
        """
        # Prepare input data
        videos_json = json.dumps(videos, ensure_ascii=False, indent=2)
        subtitles_json = json.dumps(subtitles, ensure_ascii=False, indent=2)

        print("Analyzing files with LLM...")

        # Create messages for chat template
        system_message = LlmChatMessage(
            role="system",
            content="You are an expert at matching subtitle files with video files. Analyze the file names and match them based on episode numbers or sequence patterns.",
        )

        user_message = LlmChatMessage(
            role="user",
            content=f"""Match these subtitle files with their corresponding video files.

Video files:
{videos_json}

Subtitle files:
{subtitles_json}

Look for patterns like:
- Episode numbers (e.g., E01, EP01, 第1集, [01], - 01)
- Season and episode (e.g., S01E01)
- Sequential numbering in filenames
- Similar titles or names

Return ONLY a valid JSON object mapping subtitle filename to video filename:
{{"subtitle_file_1.srt": "video_file_1.mkv", "subtitle_file_2.srt": "video_file_2.mkv"}}

Do not include any explanation or markdown formatting, just the raw JSON.""",
        )

        # Apply chat template to format the prompt
        formatted_prompt = self.llm.apply_chat_template(
            messages=[system_message, user_message], add_generation_prompt=True
        )

        print("\n" + "=" * 60)
        print("Formatted Prompt:")
        print("=" * 60)
        print(
            formatted_prompt[:500] + "..."
            if len(formatted_prompt) > 500
            else formatted_prompt
        )
        print("=" * 60 + "\n")

        # Create generation config with more tokens
        sampler_config = SamplerConfig(
            temperature=0.1,  # Lower temperature for more consistent output
            enable_json=True,  # Enable JSON mode for better formatting
        )
        gen_config = GenerationConfig(
            max_tokens=2048,  # Increase max tokens to handle all matches
            sampler_config=sampler_config,
            stop=["<|im_end|>", "</s>"],
        )

        # Generate response from LLM
        response = self.llm.generate(formatted_prompt, config=gen_config)
        response_text = (
            response.full_text if hasattr(response, "full_text") else str(response)
        )

        print("\n" + "=" * 60)
        print("LLM Response:")
        print("=" * 60)
        print(response_text)
        print("=" * 60 + "\n")

        # Clean up markdown formatting if present
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith("```"):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove trailing ```
        response_text = response_text.strip()

        # Extract JSON from response
        try:
            # Try to parse the entire response as JSON
            parsed = json.loads(response_text)

            # If it's an array of single-key objects, convert to dict
            if isinstance(parsed, list):
                matches = {}
                for item in parsed:
                    if isinstance(item, dict):
                        matches.update(item)
            else:
                matches = parsed
            print(f"✓ Successfully parsed and extracted {len(matches)} matches")
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse entire response as JSON: {e}")
            # Try to find JSON object in the response (more flexible regex)
            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text, re.DOTALL
            )
            if json_match:
                print(f"Found JSON pattern: {json_match.group()[:200]}...")
                try:
                    matches = json.loads(json_match.group())
                    print(
                        f"✓ Successfully parsed extracted JSON with {len(matches)} matches"
                    )
                except json.JSONDecodeError as e2:
                    print(f"✗ Failed to parse extracted JSON: {e2}")
                    print(
                        "Warning: Could not parse LLM response, returning empty matches"
                    )
                    matches = {}
            else:
                print("Warning: No valid JSON found in LLM response")
                matches = {}

        return matches

    def display_matches(self, matches: Dict[str, str], directory: str) -> bool:
        """Display the matched pairs and ask for user confirmation.

        Args:
            matches: Dictionary of subtitle to video mappings
            directory: Directory path for display

        Returns:
            True if user confirms, False otherwise
        """
        if not matches:
            print("No matches found!")
            return False

        print("\n" + "=" * 60)
        print("Subtitle to Video Matching Results")
        print("=" * 60)
        for i, (subtitle, video) in enumerate(matches.items(), 1):
            print(f"{i}. {subtitle:40} -> {video}")
        print("=" * 60)

        while True:
            response = input("\nConfirm these matches? (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def rename_subtitles(
        self, matches: Dict[str, str], directory: str
    ) -> List[Tuple[str, str]]:
        """Rename subtitle files to match their corresponding videos.

        Args:
            matches: Dictionary of subtitle to video mappings
            directory: Directory where files are located

        Returns:
            List of tuples (old_name, new_name) for renamed files
        """
        renamed_files = []

        for subtitle_name, video_name in matches.items():
            subtitle_path = Path(directory) / subtitle_name

            # Get the video extension and create new subtitle name
            video_without_ext = Path(video_name).stem
            subtitle_ext = Path(subtitle_name).suffix
            new_subtitle_name = video_without_ext + subtitle_ext
            new_subtitle_path = Path(directory) / new_subtitle_name

            try:
                subtitle_path.rename(new_subtitle_path)
                renamed_files.append((subtitle_name, new_subtitle_name))
                print(f"Renamed: {subtitle_name} -> {new_subtitle_name}")
            except Exception as e:
                print(f"Error renaming {subtitle_name}: {e}")

        return renamed_files


def main():
    """Main function for the subtitle renamer tool."""
    parser = argparse.ArgumentParser(
        description="Rename subtitles to match video files using LLM"
    )
    parser.add_argument(
        "-d",
        "--dir",
        default=".",
        help="Directory containing video and subtitle files (default: current directory)",
    )

    args = parser.parse_args()
    directory = os.path.abspath(args.dir)

    # Validate directory
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory")
        return

    print(f"Scanning directory: {directory}")

    renamer = SubtitleRenamer(nexaai_llm)

    # Get files
    videos, subtitles = renamer.get_files_in_directory(directory)

    if not videos:
        print("No video files found in the directory")
        return

    if not subtitles:
        print("No subtitle files found in the directory")
        return

    print(f"Found {len(videos)} video file(s) and {len(subtitles)} subtitle file(s)\n")

    # Match subtitles with videos using LLM
    print("Using LLM to match subtitles with videos...")
    matches = renamer.match_subtitles_with_llm(videos, subtitles)

    # Display and confirm matches
    if renamer.display_matches(matches, directory):
        print("\nRenaming subtitles...")
        renamed = renamer.rename_subtitles(matches, directory)
        print(f"\nSuccessfully renamed {len(renamed)} subtitle file(s)")
    else:
        print("Operation cancelled")


if __name__ == "__main__":
    main()

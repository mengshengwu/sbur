# SBUR

Intelligent subtitle file renamer using LLM-based episode matching.

## Quick Start

```bash
# Install
pip install -e .

# Use
sbur -d /path/to/media
```

## Supported Formats

**Video:** mp4, mkv, avi, mov, flv, wmv, webm  
**Subtitle:** srt, ass, ssa, sub, vtt

## Usage

```bash
sbur -d /path/to/media          # Rename subtitles in directory
sbur -d /path/to/media -v       # With debug output
sbur --help                      # Show all options
```

Example:
```
Video:    [Show] Episode [01].mkv
Subtitle: Show - 01.srt
Result:   [Show] Episode [01].srt
```

## Development

```bash
uv sync
pytest test.py -s        # Run tests with output
```

## License

MIT

## Acknowledgements

Built with [NexaAI Nexa SDK](https://github.com/NexaAI/nexa-sdk)


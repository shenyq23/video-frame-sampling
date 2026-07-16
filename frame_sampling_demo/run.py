"""Run the CLI directly from a source checkout without installing the demo."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from frame_sampling_demo.cli import main


if __name__ == "__main__":
    main()

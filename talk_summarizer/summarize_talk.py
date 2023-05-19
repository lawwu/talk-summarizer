import os
import openai
from pathlib import Path
import logging

from talk_summarizer.utils import (
    extract_audio,
    transcribe_audio,
    extract_slides,
    divide_transcript,
    get_section_transcripts,
    summarize_sections,
    generate_overall_summary,
    generate_pdf_report,
)

OUTPUT_DIR = Path("output")

openai.api_key = os.getenv("OPENAI_API_KEY")

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def main(video_file, output_dir, summarize=False, whisper_model="tiny"):
    # Create output directory if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract audio from video
    audio_file = extract_audio(video_file, "audio.mp3")

    # Transcribe audio
    transcript = transcribe_audio(
        audio_file, model=whisper_model, output_dir=output_dir
    )

    # Extract slides from video
    slide_transitions = extract_slides(video_file, output_dir)

    # Divide transcript into sections based on slide timestamps
    divided_transcript = divide_transcript(transcript, slide_transitions)

    # Get the transcript for each section
    section_transcripts = get_section_transcripts(divided_transcript)

    if summarize:
        # Summarize each section of the transcript
        summarized_sections = summarize_sections(section_transcripts)

        # Generate overall summary of the entire transcript
        overall_summary = generate_overall_summary(summarized_sections)
    else:
        summarized_sections = None
        overall_summary = None

    # Create a PDF report with slide images and summaries
    slide_images = sorted(output_dir.glob("*.png"))
    report_filename = os.path.join(output_dir, "report.pdf")
    generate_pdf_report(
        report_filename,
        overall_summary,
        slide_images,
        summarized_sections,
        section_transcripts,
        summarize=summarize,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize a lecture video")
    parser.add_argument("--video_file", help="Path to the video file")
    parser.add_argument(
        "--output_dir", default="output", help="Path to the output directory"
    )
    parser.add_argument(
        "--summarize",
        action=argparse.BooleanOptionalAction,
        help="Summarize the sections (default: False)",
    )
    parser.add_argument(
        "--whisper_model",
        default="medium",
        help="Whisper model to use (default: medium)",
    )
    args = parser.parse_args()

    main(args.video_file, args.output_dir, args.summarize, args.whisper_model)

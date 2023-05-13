# talk-summarizer

Python library to summarize talks. Turn a video file into a PDF report with a summary of each section of the talk.

`summarize_talk.py` - operates on a talk/lecture where there is a video of the speaker and slides. 

- uses `ffmpeg` to extract the audio from the video
- uses `whisper` to transcribe the audio
- uses `OpenCV` to identify slide transitions and extract slides
- uses `OpenAI` to summarize each section

## Installation

Install this library using `pip`:

    pip install talk-summarizer

## Usage

```bash
python talk_summarizer/summarize_talk.py \
    --output_dir output \
    --video_file video.mp4 \
    --no-summarize \
    --whisper_model tiny
```

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:

    cd talk-summarizer
    python -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -r requirements.txt

To run the tests:

    pytest

# Set API keys

Make sure your `OPENAI_API_KEY` is set as an environment variable in either your `.bashrc` or `.zshrc` file. For example:

```bash
export OPENAI_API_KEY=sk-1234...
```

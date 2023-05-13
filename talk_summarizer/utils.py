import ffmpeg
import whisper
import cv2
import numpy as np
import openai
from pathlib import Path
import logging
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

OUTPUT_DIR = Path("output")

# add logging
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# Extract audio from video
def extract_audio(video_file, audio_file):
    """
    Extract audio from video

    Parameters
    ----------
    video_file : str
        Path to video file
    audio_file : str
        Path to audio file

    Returns
    -------
    None
    """
    logging.info(f"Using ffmpeg to extract audio from {video_file} to {audio_file}")
    try:
        (
            ffmpeg
            .input(video_file)
            .output(audio_file, acodec='mp3', ac=2, ar='48k', ab='192k')
            .run(overwrite_output=True)
        )
        return audio_file
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

# Transcribe audio using whisper
def transcribe_audio(audio_file, output_dir, model="tiny"):
    """
    Transcribe audio using whisper
    """
    logging.info(f"Transcribing audio with whisper model: {model}")
    model = whisper.load_model(model)
    result = model.transcribe(audio_file)

    # write result to JSON file
    with open(output_dir / "transcript.json", "w") as f:
        json.dump(result, f)

    return result


def frame_difference(frame1, frame2):
    """
    Calculate the difference between two frames
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return np.sum(cv2.absdiff(gray1, gray2))


def extract_slides(video_file, 
                   output_dir, 
                   threshold=3_000_000, 
                   min_seconds=2):
    """
    Extract slides from video file into output directory
    """

    logging.info("Extracting slides from video file into output directory")
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    slide_transitions = [(0, prev_frame)]

    # Save the first slide
    cv2.imwrite(f"{output_dir}/slide_0001.png", prev_frame)

    for i in range(1, frame_count):
        ret, curr_frame = cap.read()
        if not ret:
            break

        diff = frame_difference(prev_frame, curr_frame)
        timestamp = i / fps

        # Check if the difference is above the threshold and more than 2 seconds have passed since the last transition
        if diff > threshold and (timestamp - slide_transitions[-1][0]) > min_seconds:
            slide_transitions.append((timestamp, curr_frame))
            cv2.imwrite(f"{output_dir}/slide_{len(slide_transitions):04d}.png", curr_frame)

        prev_frame = curr_frame

    cap.release()
    
    # log how many frames captured
    logging.info(f"Extracted {len(slide_transitions)} slides from {video_file}")

    return slide_transitions


def divide_transcript(transcript, slide_transitions):
    logging.info("Dividing transcript into sections...")
    slide_timestamps = [t[0] for t in slide_transitions]

    # Add an extra timestamp to account for the end of the video
    slide_timestamps.append(float('inf'))

    divided_transcript = []
    current_slide = 0
    current_slide_segments = []

    for segment in transcript['segments']:
        segment_start = segment['start']
        # segment_end = segment['end']

        while segment_start >= slide_timestamps[current_slide + 1]:
            # If the segment starts after the current slide ends, finalize the current slide
            divided_transcript.append(current_slide_segments)
            current_slide += 1
            current_slide_segments = []

        current_slide_segments.append(segment)

    # Append the last slide's segments
    divided_transcript.append(current_slide_segments)

    return divided_transcript


def get_section_transcripts(divided_transcript):
    section_transcripts = []

    for section_segments in divided_transcript:
        section_text = " ".join(segment['text'] for segment in section_segments)
        section_transcripts.append(section_text)

    return section_transcripts


def chunk_text(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Summarize each section of the transcript
def summarize_sections(section_transcripts, 
                       model="gpt-3.5-turbo", 
                       prompt_template="Please summarize the following text: '{}'",
                       temperature=0,
                       stream=False,
                       max_input_tokens=4096,
                       ):
    logging.info(f"Summarizing each section of transcript, there are {len(section_transcripts)} total.")
    
    summaries = []

    for section_text in section_transcripts:
        # print(section_text)
        section_chunks = chunk_text(section_text, max_input_tokens)
        section_summary = ""

        for chunk in section_chunks:
            prompt = prompt_template.format(chunk)
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an A+ student and a great summarizer of lecture notes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=stream,
            )

            summary = response.choices[0].message.content
            section_summary += " " + summary

        summaries.append(section_summary.strip())

    return summaries


def generate_overall_summary(section_summaries,
                             model="gpt-3.5-turbo",
                             prompt_template="Please provide an overall summary for the following summaries: '{}'",
                             temperature=0,
                             stream=False
                             ):
    logging.info("Generating overall summary of the entire transcript...")    
    summaries_text = " ".join(section_summaries)
    prompt = prompt_template.format(summaries_text)
    response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an A+ student and a great summarizer of lecture notes.",
                },
                {"role": "user", "content": prompt},
            ],
        temperature=temperature,
        stream=stream,
        )

    overall_summary = response.choices[0].message.content
    return overall_summary


# Create a PDF report with slide images and summaries
def generate_pdf_report(output_filename, 
                        overall_summary, 
                        slide_images, 
                        section_summaries, 
                        section_transcripts,
                        summarize=False
                        ):
    logging.info("Creating a PDF report with slide images and summaries...")
    
    doc = SimpleDocTemplate(output_filename, pagesize=letter)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))

    story = []

    # Overall summary
    if summarize:
        story.append(Paragraph("Overall Summary", styles['Heading1']))
        story.append(Paragraph(overall_summary, styles['BodyText']))
        story.append(Spacer(1, 12))

        # page break
        story.append(PageBreak())

    if summarize:
        for i, (slide_image_path, section_summary, section_transcript) in enumerate(zip(slide_images, section_summaries, section_transcripts)):
            # Slide image
            story.append(Paragraph(f"Slide {i + 1}", styles['Heading2']))
            slide_image = Image(slide_image_path, width=500, height=375)
            story.append(slide_image)
            story.append(Spacer(1, 12))

            # Section summary
            story.append(Paragraph("Section Summary", styles['Heading3']))
            story.append(Paragraph(section_summary, styles['BodyText']))
            story.append(Spacer(1, 12))

            # Transcript of the section
            story.append(Paragraph("Section Transcript", styles['Heading3']))
            story.append(Paragraph(section_transcript, styles['BodyText']))
            story.append(Spacer(1, 12))

            # Page break
            story.append(PageBreak())
    else: 
        for i, (slide_image_path, section_transcript) in enumerate(zip(slide_images, section_transcripts)):
            # Slide image
            story.append(Paragraph(f"Slide {i + 1}", styles['Heading2']))
            slide_image = Image(slide_image_path, width=500, height=375)
            story.append(slide_image)
            story.append(Spacer(1, 12))

            # Transcript of the section
            story.append(Paragraph("Section Transcript", styles['Heading3']))
            story.append(Paragraph(section_transcript, styles['BodyText']))
            story.append(Spacer(1, 12))

            # Page break
            story.append(PageBreak())   

    doc.build(story)

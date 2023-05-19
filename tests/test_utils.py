import numpy as np
from talk_summarizer.utils import (
    frame_difference,
    chunk_text,
)


def test_frame_difference():
    # Create two test frames
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Calculate the difference between the frames
    diff = frame_difference(frame1, frame2)

    # Check that the difference is as expected
    expected_diff = 100 * 100 * 255
    assert diff == expected_diff


def test_chunk_text():
    text = "This is another test sentence with a longer word."
    max_tokens = 5
    expected_output = [
        "This",
        "is",
        "another",
        "test",
        "sentence",
        "with",
        "a",
        "longer",
        "word.",
    ]
    assert chunk_text(text, max_tokens) == expected_output

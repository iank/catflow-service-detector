from catflow_service_detector.worker import get_most_detected_class
from pytest import approx
from catflow_worker.types import (
    AnnotatedFrame,
    VideoFile,
    Prediction,
)


def test_get_most_detected_class():
    # We want to get the class that is seen in the most frames, as well as each
    # frame/detection. Two details:
    #
    # If class Y is seen 10 times in 1 frame but 10 times overall, and class X
    # is seen once in 4 frames each, we want class X.
    #
    # If class X is the most-detected-class, and it's seen twice in one frame, just
    # pick the detection w/ the highest confidence for that frame.
    frames = [
        AnnotatedFrame(
            key="frame1.png",
            source=VideoFile(key="test.mp4"),
            model_name="test_model",
            predictions=[
                Prediction(x=1, y=2, width=4, height=10, confidence=0.97, label="X"),
                Prediction(x=5, y=6, width=8, height=12, confidence=0.50, label="X"),
            ],
        ),
        AnnotatedFrame(
            key="frame2.png",
            source=VideoFile(key="test.mp4"),
            model_name="test_model",
            predictions=[
                Prediction(x=3, y=4, width=5, height=6, confidence=0.77, label="Y"),
                Prediction(x=7, y=8, width=9, height=10, confidence=0.53, label="X"),
            ],
        ),
        AnnotatedFrame(
            key="frame3.png",
            source=VideoFile(key="test.mp4"),
            model_name="test_model",
            predictions=[
                Prediction(x=5, y=6, width=7, height=8, confidence=0.42, label="X"),
                Prediction(x=9, y=10, width=11, height=12, confidence=0.30, label="Z"),
            ],
        ),
        AnnotatedFrame(
            key="frame4.png",
            source=VideoFile(key="test.mp4"),
            model_name="test_model",
            predictions=[
                Prediction(x=1, y=2, width=3, height=4, confidence=0.85, label="X"),
                Prediction(x=5, y=6, width=7, height=8, confidence=0.23, label="Y"),
                Prediction(x=9, y=10, width=11, height=12, confidence=0.24, label="Y"),
                Prediction(x=13, y=14, width=15, height=16, confidence=0.28, label="Y"),
                Prediction(x=17, y=18, width=19, height=20, confidence=0.39, label="Y"),
                Prediction(x=21, y=22, width=23, height=24, confidence=0.99, label="Y"),
                Prediction(x=25, y=26, width=27, height=28, confidence=0.87, label="Y"),
                Prediction(x=29, y=30, width=31, height=32, confidence=0.20, label="Y"),
            ],
        ),
        AnnotatedFrame(
            key="frame5.png",
            source=VideoFile(key="test.mp4"),
            model_name="test_model",
            predictions=[
                Prediction(x=1, y=2, width=3, height=4, confidence=0.80, label="Z"),
            ],
        ),
    ]

    most_detected_class, detected_frames = get_most_detected_class(
        frames, ["X", "Y", "Z"]
    )

    assert most_detected_class == "X"
    assert len(detected_frames) == 4

    for frame in detected_frames:
        assert len(frame.predictions) == 1
        assert frame.predictions[0].label == most_detected_class

    assert detected_frames[0].predictions[0].confidence == approx(0.97)


def test_get_most_detected_class_filter():
    # If we ask for X and Y, but Z is also present, we just want X or Y
    frames = [
        AnnotatedFrame(
            key="frame1.png",
            source=VideoFile(key="test.mp4"),
            model_name="test_model",
            predictions=[
                Prediction(x=1, y=2, width=4, height=10, confidence=0.97, label="X"),
                Prediction(x=5, y=6, width=8, height=12, confidence=0.50, label="X"),
                Prediction(x=9, y=10, width=11, height=12, confidence=0.30, label="Z"),
            ],
        ),
        AnnotatedFrame(
            key="frame2.png",
            source=VideoFile(key="test.mp4"),
            model_name="test_model",
            predictions=[
                Prediction(x=3, y=4, width=5, height=6, confidence=0.77, label="Y"),
                Prediction(x=7, y=8, width=9, height=10, confidence=0.53, label="X"),
                Prediction(x=9, y=10, width=11, height=12, confidence=0.30, label="Z"),
            ],
        ),
        AnnotatedFrame(
            key="frame3.png",
            source=VideoFile(key="test.mp4"),
            model_name="test_model",
            predictions=[
                Prediction(x=5, y=6, width=7, height=8, confidence=0.42, label="X"),
                Prediction(x=9, y=10, width=11, height=12, confidence=0.30, label="Z"),
            ],
        ),
        AnnotatedFrame(
            key="frame4.png",
            source=VideoFile(key="test.mp4"),
            model_name="test_model",
            predictions=[
                Prediction(x=1, y=2, width=3, height=4, confidence=0.85, label="X"),
                Prediction(x=9, y=10, width=11, height=12, confidence=0.24, label="Y"),
                Prediction(x=9, y=10, width=11, height=12, confidence=0.30, label="Z"),
            ],
        ),
    ]

    most_detected_class, detected_frames = get_most_detected_class(frames, ["X", "Y"])

    assert most_detected_class == "X"

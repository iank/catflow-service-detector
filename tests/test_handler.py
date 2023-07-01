import pytest
from catflow_service_detector.worker import create_detector_handler
from moto import mock_s3
import boto3
import io
from catflow_worker.types import (
    AnnotatedFrame,
    AnnotatedFrameSchema,
    VideoFile,
    Prediction,
)


@pytest.fixture()
def notifier():
    return Notifier_Mock()


class Notifier_Mock:
    """Fake notification handler"""

    def __init__(self):
        self.notifications = []

    async def notify(self, image, label):
        self.notifications.append((image, label))


AWS_BUCKET_NAME = "test-bucket"


class AsyncS3Wrapper:
    """Just fake it so I can still mock with moto

    The alternative is setting up a mock server and connecting to it, as in
    catflow-worker's tests"""

    def __init__(self):
        session = boto3.Session()
        self.client = session.client("s3", region_name="us-east-1")

    async def download_fileobj(self, Bucket=None, Key=None, Fileobj=None):
        buf = io.BytesIO()
        self.client.download_fileobj(Bucket, Key, buf)

        # Ensure we're reading from the start of the file
        buf.seek(0)
        data = buf.read()

        Fileobj.write(data)

        return Fileobj


@pytest.fixture(scope="session")
def s3_client():
    # Get data file
    image = "tests/test_files/car.png"

    with mock_s3():
        s3 = AsyncS3Wrapper()

        # Push it to mock S3 so our worker can retrieve it
        s3.client.create_bucket(Bucket=AWS_BUCKET_NAME)
        with open(image, "rb") as f:
            s3.client.upload_fileobj(f, AWS_BUCKET_NAME, "test1.png")
        with open(image, "rb") as f:
            s3.client.upload_fileobj(f, AWS_BUCKET_NAME, "test2.png")
        with open(image, "rb") as f:
            s3.client.upload_fileobj(f, AWS_BUCKET_NAME, "test3.png")
        with open(image, "rb") as f:
            s3.client.upload_fileobj(f, AWS_BUCKET_NAME, "test4.png")

        yield s3


@pytest.mark.asyncio
async def test_worker(notifier, s3_client):
    detector_handler = create_detector_handler(notifier, ["cat", "dog"])

    # Expected input: list of AnnotatedFrame
    video = VideoFile(key="test.mp4")
    frames = [
        AnnotatedFrame(
            key="test1.png",
            source=video,
            model_name="test",
            predictions=[
                Prediction(x=1, y=2, width=4, height=10, confidence=0.97, label="cat")
            ],
        ),
        AnnotatedFrame(
            key="test2.png",
            source=video,
            model_name="test",
            predictions=[
                Prediction(x=1, y=2, width=4, height=10, confidence=0.97, label="cat")
            ],
        ),
        AnnotatedFrame(
            key="test3.png",
            source=video,
            model_name="test",
            predictions=[
                Prediction(x=1, y=2, width=4, height=10, confidence=0.97, label="dog")
            ],
        ),
        AnnotatedFrame(
            key="test4.png",
            source=video,
            model_name="test",
            predictions=[
                Prediction(x=1, y=2, width=4, height=10, confidence=0.97, label="cat")
            ],
        ),
    ]
    annotations_msg = AnnotatedFrameSchema(many=True).dump(frames)

    # Expected output: none
    status, responses = await detector_handler(
        annotations_msg, "detect.annotatedframes", s3_client, AWS_BUCKET_NAME
    )

    assert status is True
    assert len(responses) == 0

    # check notifier
    assert len(notifier.notifications) == 1
    assert notifier.notifications[0][1] == "cat"

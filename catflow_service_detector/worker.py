from typing import Any, List, Tuple
import signal
import asyncio
from catflow_worker import Worker
from catflow_worker.types import (
    AnnotatedFrame,
    AnnotatedFrameSchema,
)
from collections import Counter
from PIL import Image

import os
import io
import cv2
import aiohttp
import base64
import scipy
import datetime
import numpy as np

import logging


def get_most_detected_class(frames: AnnotatedFrame) -> Tuple[str, AnnotatedFrame]:
    # Find the most detected class
    detections = Counter()
    for frame in frames:
        detected_labels = set([x.label for x in frame.predictions])
        for label in detected_labels:
            detections[label] += 1

    if len(detections) == 0:
        return None, []

    most_detected_class = detections.most_common(1)[0][0]

    # Get the highest confidence detection for the most detected class from each frame
    detected_frames = []
    for frame in frames:
        detections = [x for x in frame.predictions if x.label == most_detected_class]
        if len(detections) == 0:
            continue

        most_confident_detection = max(detections, key=lambda x: x.confidence)
        detected_frame = AnnotatedFrame(
            key=frame.key,
            source=frame.source,
            model_name=frame.model_name,
            predictions=[most_confident_detection],
        )
        detected_frames.append(detected_frame)

    return most_detected_class, detected_frames


def draw_detection(frame, predictions):
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    imgmask = np.ones(image.shape[:2])
    y_coords, x_coords = np.ogrid[: image.shape[0], : image.shape[1]]

    # Draw the detections
    for p in predictions:
        dist_from_center = ((x_coords - p.x) / p.width) ** 2 + (
            (y_coords - p.y) / p.height
        ) ** 2
        mask = dist_from_center <= 0.5
        imgmask[mask] = 0

    sigma = 20
    imgmask = scipy.ndimage.gaussian_filter(imgmask, sigma)
    imgmask = 1 - imgmask
    imgmask = imgmask * 0.85 + 0.15
    image[:, :, 2] = (image[:, :, 2] * imgmask).astype("uint8")

    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    # Encode
    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()
    return img_bytes


class Notifier:
    """Notification handler"""

    def __init__(self, base_url, source, dest):
        self.base_url = base_url
        self.source = source
        self.dest = dest

    async def notify(self, image, label):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_base64 = base64.b64encode(image).decode()

        json_payload = {
            "message": f"Detected {label} at {timestamp}",
            "base64_attachments": [image_base64],
            "number": self.source,
            "recipients": [self.dest],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v2/send",
                json=json_payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                logging.info(f"Response status: {resp.status}")
                logging.info(await resp.text())


def create_detector_handler(notifier, classes_of_interest):
    async def _detector_handler(
        msg: Any, key: str, s3: Any, bucket: str
    ) -> Tuple[bool, List[Tuple[str, Any]]]:
        """Run detector on the given frames"""
        logging.info(f"[*] Message received ({key})")

        # Load message
        annotated_frames = AnnotatedFrameSchema(many=True).load(msg)

        # Filter frames for classes of interest
        interest_frames = []
        for frame in annotated_frames:
            predicted_labels = [x.label for x in frame.predictions]
            intersection = set(predicted_labels).intersection(set(classes_of_interest))
            if len(intersection):
                interest_frames.append(frame)

        # Get the class w/ the most detections, and the frames it was detected in.
        # detected_frames is a list of AnnotatedFrame, now with exactly 1 prediction per
        # frame.
        most_detected_class, detected_frames = get_most_detected_class(interest_frames)

        logging.info(
            "[-] Detected {l} in {n} of {N} frames".format(
                l=most_detected_class, n=len(detected_frames), N=len(annotated_frames)
            )
        )

        if len(detected_frames) == 0:
            return True, []

        # Pick the middle sighting
        picked_frame = detected_frames[len(detected_frames) // 2]

        # Download the frame
        frame_file = io.BytesIO()
        await s3.download_fileobj(bucket, picked_frame.key, frame_file)
        frame_file.seek(0)

        # Draw on it
        image = Image.open(frame_file)
        image_drawn = draw_detection(
            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), picked_frame.predictions
        )

        # Notify
        await notifier.notify(image_drawn, most_detected_class)

        return True, []

    return _detector_handler


async def shutdown(worker, task):
    await worker.shutdown()
    task.cancel()
    try:
        await task
    except asyncio.exceptions.CancelledError:
        pass


async def startup(queue: str, topic_key: str):
    # Set up DB
    base_url = os.environ["CATFLOW_SIGNAL_URL"]
    source = os.environ["CATFLOW_SIGNAL_SOURCE"]
    dest = os.environ["CATFLOW_SIGNAL_DEST"]
    classes = os.environ["CATFLOW_DETECTOR_CLASSES"].split(",")

    notifier = Notifier(base_url, source, dest)
    detector_handler = create_detector_handler(notifier, classes)

    # Start worker
    worker = await Worker.create(detector_handler, queue, topic_key)
    task = asyncio.create_task(worker.work())

    def handle_sigint(sig, frame):
        print("^ SIGINT received, shutting down...")
        asyncio.create_task(shutdown(worker, task))

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        if not await task:
            print("[!] Exited with error")
            return False
    except asyncio.exceptions.CancelledError:
        return True


def main() -> bool:
    topic_key = "detect.annotatedframes"
    queue_name = "catflow-service-detector"
    logging.basicConfig(level=logging.INFO)

    return asyncio.run(startup(queue_name, topic_key))

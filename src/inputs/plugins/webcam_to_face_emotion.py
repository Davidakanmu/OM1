import asyncio
import time
from dataclasses import dataclass

from typing import Optional

from inputs.base.loop import LoopInput

from providers.io_provider import IOProvider

import cv2
from deepface import DeepFace

@dataclass
class Message:
    timestamp: float
    message: str

"""
Code example is from:
https://github.com/manish-9245/Facial-Emotion-Recognition-using-OpenCV-and-Deepface
Thank you @manish-9245
"""

class FaceEmotionCapture(LoopInput[cv2.typing.MatLike]):
    """
    Uses a webcam and returns a label of the person's emotions
    """

    def __init__(self):
        # Track IO
        self.io_provider = IOProvider()

        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Start capturing video
        self.cap = cv2.VideoCapture(0)

        # Initialize emotion label
        self.emotion = ""

        # Messages buffer
        self.messages: list[Message] = []

    async def _poll(self) -> cv2.typing.MatLike:
        await asyncio.sleep(0.2)

        # Capture a frame every 200 ms
        ret, frame = self.cap.read()

        return frame

    async def _raw_to_text(self, raw_input: cv2.typing.MatLike) -> Message:

        frame = raw_input

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            self.emotion = result[0]['dominant_emotion']

        message = f"I see a person. Their emotion is {self.emotion}."

        return Message(timestamp=time.time(), message=message)

    async def raw_to_text(self, raw_input):
        """
        Convert raw input to processed text and manage buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
        {self.__class__.__name__} INPUT
        // START
        {latest_message.timestamp:.3f}
        // END
        """

        self.io_provider.add_input(self.__class__.__name__, latest_message.message, latest_message.timestamp)
        self.messages = []

        return result

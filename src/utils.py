import cv2
import numpy as np

class Video:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.file_path)
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def stack_frames(self, start, total_frames, gray=True):
        self.cap.set(1, start)

        count = 0
        while count < total_frames:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("frame cannot be read!")
            if gray:
                frame = Video.convert_grayscale(frame)
            frame.

            count = count + 1

    @staticmethod
    def convert_grayscale(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


#creates row major tall skinny vector from a list of numpy nd array
def create_tall_skinny(X):
    return X.flatten(order = 'C')

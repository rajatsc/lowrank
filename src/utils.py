import cv2
import numpy as np

class VideoIO:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cap = cv2.VideoCapture(self.file_path)

    def teardown(self):
        if self.cap:
            self.cap.release()

    def stack_frames(self, start, total_frames, gray=True):
        self.cap.set(1, start)

        count = 0
        frames_list = []
        while count < total_frames:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("frame cannot be read!")
            if gray:
                frame = VideoIO.convert_grayscale(frame)
                frame = cv2.pyrDown(frame)
                frame = cv2.pyrDown(frame)
            frames_list.append(frame)
            count = count + 1

        return np.asarray(frames_list)

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @staticmethod
    def convert_grayscale(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

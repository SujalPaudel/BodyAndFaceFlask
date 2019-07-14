import numpy as np
import argparse
import imutils
import time
import cv2
from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
import glob
import os
import time
import cv2
import imutils
from imutils.object_detection import non_max_suppression


font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
recognizer = cv2.face.LBPHFaceRecognizer_create()
count = 0

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        self.video = FileVideoStream("cropvideo.mp4").start()
    
    def __del__(self):
        self.video.release()

    def detect_people(self, frame):

        (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.06)
        rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return frame


    def detect_face(self, frame):

        faces = face_cascade.detectMultiScale(frame, 1.1, 2, 0, (20, 20))
        return faces


    def draw_faces(self, frame, faces):

        for (x, y, w, h) in faces:
            xA = x
            yA = y
            xB = x + w
            yB = y + h
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        return frame


    def background_subtraction(self, previous_frame, frame_resized_grayscale, min_area):

        frameDelta = cv2.absdiff(previous_frame, frame_resized_grayscale)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        temp = 0
        for c in cnts:
            if cv2.contourArea(c) > min_area:
                temp = 1
        return temp


    
    def get_frame(self):

        frame = self.video.read()

        frame_resized_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_processed = self.detect_people(frame)

        faces = self.detect_face(frame)

        frame_processed = self.draw_faces(frame_processed, faces)

        ret, jpeg = cv2.imencode('.jpg', frame_processed)

        return jpeg.tobytes()

       


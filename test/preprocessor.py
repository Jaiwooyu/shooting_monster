import os
import cv2
import numpy as np
import torch
from .utils import calculate_angle, check_shot_form_quality
import json

class ShootingFormPreprocessor:
    def __init__(self):
        self.datum, self.opWrapper = self.initialize_openpose()
        
    def initialize_openpose(self):
        try:
            if platform == "win32":
                sys.path.append(os.path.dirname(os.getcwd()))
                import OpenPose.Release.pyopenpose as op
            else:
                sys.path.append('/usr/local/python')
                import pyopenpose as op
            
            params = dict()
            params["model_folder"] = "./OpenPose/models"
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            return op.Datum(), opWrapper
            
        except Exception as e:
            print(f"Error initializing OpenPose: {e}")
            raise

    def process_sequence(self, sequence_path):
        keypoints_sequence = []
        quality_scores = []
        
        for frame_path in sorted(os.listdir(sequence_path)):
            if frame_path.endswith('.jpg'):
                frame = cv2.imread(os.path.join(sequence_path, frame_path))
                keypoints = self.extract_keypoints(frame)
                if keypoints is not None:
                    keypoints_sequence.append(keypoints)
                    quality_scores.append(check_shot_form_quality(keypoints))
                    
        return np.array(keypoints_sequence), np.array(quality_scores)

    def extract_keypoints(self, frame):
        self.datum.cvInputData = frame
        self.opWrapper.emplaceAndPop([self.datum])
        
        try:
            return self.datum.poseKeypoints[0]
        except:
            return None
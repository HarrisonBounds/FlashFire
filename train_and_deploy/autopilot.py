
# Deploy trained neural network for self driving 


#!/usr/bin/python3
import sys
import os
import cv2 as cv
from gpiozero import Servo, PhaseEnableMotor

from time import time
import torch
import torch.nn as nn
from torchvision import transforms
import cnn_network

num_parameters = 2
if len(sys.argv) != num_parameters:
    print(f'Python script needs {num_parameters} parameters!!!')
else:
    model_name = sys.argv[1]
    
# SETUP
# load configs
# init servo controller
model_path = os.path.join(sys.path[0], 'models', model_name)
to_tensor = transforms.ToTensor()
model = cnn_network.DonkeyNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# init variables
throttle, steer = 0., 0.
is_recording = False
frame_counts = 0
# init camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 20)
for i in reversed(range(60)):  # warm up camera
    if not i % 20:
        print(i/20)
    ret, frame = cap.read()
# init timer, uncomment if you are cuious about frame rate
start_stamp = time()
ave_frame_rate = 0.

motor = PhaseEnableMotor(phase=19, enable=26)
servo = Servo(24)

# MAIN
try:
    while True:
        ret, frame = cap.read()
        if frame is not None:
            frame_counts += 1
        else:
            cv.destroyAllWindows()
            sys.exit()
        # predict steer and throttle
        image = cv.resize(frame, (200, 200))
        img_tensor = to_tensor(image)
        pred_steer, pred_throttle = model(img_tensor[None, :]).squeeze()
        steer = float(pred_steer)
        throttle = float(pred_throttle)
        # if throttle >= 1:  # predicted throttle may over the limit
        #     throttle = .999
        # elif throttle <= -1:
        #     throttle = -.999
        motor.forward(throttle)
        servo.value = steer
        action = [steer, throttle]
        print(f"action: {action}")
        # monitor frame rate
        duration_since_start = time() - start_stamp
        ave_frame_rate = frame_counts / duration_since_start
        #print(f"frame rate: {ave_frame_rate}")
        if cv.waitKey(1)==ord('q'):
            cv.destroyAllWindows()
            sys.exit()
except KeyboardInterrupt:
    cv.destroyAllWindows()
    sys.exit()

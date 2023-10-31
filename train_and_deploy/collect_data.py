
# Collect training data using bluetooth controller
# While driving, save images and joystick inputs

#!/usr/bin/python3
import sys
import os
import cv2 as cv
from gpiozero import PhaseEnableMotor
from gpiozero import Servo
import pygame
import csv
from datetime import datetime

from time import time

# SETUP
# dummy video driver
os.environ["SDL_VIDEODRIVER"] = "dummy"
# load configs
# config_path = os.path.join(sys.path[0], "config.json")
# f = open(config_path)
# data = json.load(f)
# steering_trim = -1 * data['steering_trim']
# throttle_lim = data['throttle_lim']
# init servo controller
image_dir = os.path.join(sys.path[0], 'data', datetime.now().strftime("%Y_%m_%d_%H_%M"), 'images/')
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')
# init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

# init variables
throttle, steer = 0., 0.
is_recording = False
frame_counts = 0

# init camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 20)
# for i in reversed(range(60)):  # warm up camera
#     if not i % 20:
#         print(i/20)
#     ret, frame = cap.read()
# # init timer, uncomment if you are cuious about frame rate
start_stamp = time()
ave_frame_rate = 0.
start_time=datetime.now().strftime("%Y_%m_%d_%H_%M_")

#Initialize motor and servo objects
motor = PhaseEnableMotor(phase=19, enable=26)
servo = Servo(24)
center = 0.25
offset = 0.3

servo.value = center #start servo with wheels straight

# MAIN
try:
    while True:
        ret, frame = cap.read()
        if frame is not None:
            frame_counts += 1
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                throttle = (-js.get_axis(1)) * 0.9 # throttle input: -1: max forward, 1: max backward
                steer = -js.get_axis(2)  # steer_input: -1: left, 1: right
            # elif e.type == pygame.JOYBUTTONDOWN:
            #     if pygame.joystick.Joystick(0).get_button(0):
            #         is_recording = not is_recording
            #         if is_recording:
            #             print("Recording data")
            #         else:
            #             print("Stopping data logging")
        if throttle > 0:
            motor.forward(throttle)
        elif throttle < 0:
            motor.backward(-throttle)

        if steer == 0:
            servo.value = center
        elif steer > center + offset:
            servo.value = offset
        elif steer < -(steer + offset):
            servo.value = -offset
        else:
            servo.value = steer

        action = [steer, throttle]
        #print(f"action: {action}")
        if is_recording:
            frame = cv.resize(frame, (200, 200))
            cv.imwrite(image_dir + str(frame_counts)+'.jpg', frame) # changed frame to gray
            # save labels
            label = [str(frame_counts)+'.jpg'] + action
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(label)  # write the data

        # monitor frame rate
        duration_since_start = time() - start_stamp
        ave_frame_rate = frame_counts / duration_since_start
        #print(f"frame rate: {ave_frame_rate}")

        if cv.waitKey(1)==ord('q'):
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()
except KeyboardInterrupt:
    cv.destroyAllWindows()
    pygame.quit()
    sys.exit()

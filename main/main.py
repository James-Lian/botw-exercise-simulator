# code to gather computer vision + pose data to train machine learning model
import traceback
import time
import math

import multiprocessing as multip
from multiprocessing import Manager
from collections import deque
from collections import Counter
from collections import defaultdict

from Vector3 import Vector3
from lstm_model import *

import numpy as np
import cv2
import mediapipe as mp

import serial

delay_LSTM = 0.0667


def return_landmark(landmark):
    return [landmark.x, landmark.y, landmark.z]

def capture_video(legs_buffer, arms_buffer, legs_results_buffer, arms_results_buffer, stop_e):
    # OpenCV
    FONT_BOLD = 2
    FONT_COLOUR = (0, 0, 0)
    
    # MediaPipe drawing tools and model setup
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.pose
    model = mp_holistic.Pose(model_complexity=0)
    
    # contains all joint data for the current frame
    all_joints = {}

    # LSTM
    legs_frame_deque = deque(maxlen=20)
    arms_frame_deque = deque(maxlen=15)

    cap = cv2.VideoCapture(0)
    ptime = 0
    while cap.isOpened() and not stop_e.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = time.time()

        h, w, _channels = frame.shape

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        model_results = model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        if model_results.pose_world_landmarks:
            landmarks = model_results.pose_world_landmarks.landmark
        else:
            continue

        # face
        all_joints["nose"] = return_landmark(landmarks[mp_holistic.PoseLandmark.NOSE.value])
        all_joints["l_ear"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value])
        all_joints["r_ear"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value])
        all_joints["l_eye_inner"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_EYE_INNER.value])
        all_joints["r_eye_inner"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_INNER.value])
        all_joints["l_eye"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_EYE.value])
        all_joints["r_eye"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_EYE.value])
        all_joints["l_eye_outer"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value])
        all_joints["r_eye_outer"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value])
        all_joints["l_mouth"] = return_landmark(landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value])
        all_joints["r_mouth"] = return_landmark(landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value])

        # legs
        all_joints["l_hip"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value])
        all_joints["r_hip"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value])
        all_joints["l_knee"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value])
        all_joints["r_knee"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value])
        all_joints["l_ankle"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value])
        all_joints["r_ankle"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value])
        all_joints["l_heel"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_HEEL.value])
        all_joints["r_heel"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_HEEL.value])
        all_joints["l_toe"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value])
        all_joints["r_toe"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value])

        # arms
        all_joints["l_shoulder"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value])
        all_joints["r_shoulder"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value])
        all_joints["l_elbow"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value])
        all_joints["r_elbow"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value])
        all_joints["l_wrist"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value])
        all_joints["r_wrist"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value])
        all_joints["l_thumb"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_THUMB.value])
        all_joints["r_thumb"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_THUMB.value])
        all_joints["l_index"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_INDEX.value])
        all_joints["r_index"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX.value])
        all_joints["l_pinky"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_PINKY.value])
        all_joints["r_pinky"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_PINKY.value])

        if model_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                model_results.pose_landmarks, 
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        image = cv2.flip(image, 1)
        
        ## Formatting data for LSTMs
        curr_frame_legs = [ctime] # data for current frame for legs LSTM
        curr_frame_arms = [ctime] # data for current frame for arms LSTM
        
        joints_to_remove = []
        just_legs = all_joints.copy()
        for joint in just_legs:
            if joint not in ["l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel", "r_heel", "l_toe", "r_toe"]:
                joints_to_remove.append(joint)
        for joint in joints_to_remove:
            del just_legs[joint]
        
        joints_to_remove = []
        just_arms = all_joints.copy()
        for joint in just_arms:
            if joint not in ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_thumb", "r_thumb", "l_index", "r_index", "l_pinky", "r_pinky"]:
                joints_to_remove.append(joint)
        for joint in joints_to_remove:
            del just_arms[joint]

        xlist = [value for sublist in just_legs.values() for value in sublist]
        curr_frame_legs.extend(xlist)
        
        ylist = [value for sublist in just_arms.values() for value in sublist]
        curr_frame_arms.extend(ylist)

        legs_frame_deque.append(curr_frame_legs)
        arms_frame_deque.append(curr_frame_arms)
        

        # MultiProcessing
        legs_buffer[:] = list(legs_frame_deque) # writing to the actual list, not rewriting the reference
        arms_buffer[:] = list(arms_frame_deque)

        if len(legs_results_buffer) > 0:
            leg_action_class, leg_action_index = legs_results_buffer[0]
            leg_probabilities = legs_results_buffer[1]
            probability = leg_probabilities[leg_action_index]
            
            hertz = legs_results_buffer[2]
            cv2.putText(image, leg_action_class, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, FONT_COLOUR, FONT_BOLD)
            cv2.putText(image, str(round(probability*100, 2)) + "% | " + str(hertz) + " Hz", (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_COLOUR, FONT_BOLD)


        if len(arms_results_buffer) > 0:
            arm_action_class, arm_action_index = arms_results_buffer[0]
            arm_probabilities = arms_results_buffer[1]
            probability = arm_probabilities[arm_action_index]
            
            hertz = arms_results_buffer[2]
            cv2.putText(image, arm_action_class, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, FONT_COLOUR, FONT_BOLD)
            cv2.putText(image, str(round(probability*100, 2)) + "% | " + str(hertz) + " Hz", (20, 218), cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_COLOUR, FONT_BOLD)

        # display FPS
        cv2.putText(image, "FPS: " + str(int(fps)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_COLOUR, FONT_BOLD)

        cv2.imshow("feed", image)

        key = cv2.waitKeyEx(1)
        
    cv2.destroyAllWindows()

def legs_lstm(data_buffer, results_buffer, stop_e):
    legs_lstm = ActionRecognitionModel() # 20 frames
    legs_lstm.load_model("LegsRecog_v3_best.keras", label_filepath="LegsRecog_v3.pkl")
    legs_classes = legs_lstm.label_encoder.classes_

    ptime = 0
    while True and not stop_e.is_set():
        ctime = time.time()
        hertz = 1 / (ctime - ptime)
        ptime = time.time()

        frame_folder = data_buffer[:]

        if len(frame_folder) == 20:
            start_time = frame_folder[0][0]
            for row_i in range(0, len(frame_folder)):
                frame_folder[row_i][0] -= start_time

            frame_folder = np.array(frame_folder)
            frame_folder = np.expand_dims(frame_folder, axis=0)
            action_class, probabilities = legs_lstm.predict(frame_folder)
            action_class = str(action_class[0])
            probabilities = probabilities[0].tolist()
            results_buffer[:] = [[action_class, list(legs_classes).index(action_class)], probabilities, hertz]

        time.sleep(delay_LSTM)

def arms_lstm(data_buffer, results_buffer, stop_e):
    arms_lstm = ActionRecognitionModel() # 15 frames
    arms_lstm.load_model("ArmsRecog_v2_best.keras", label_filepath="ArmsRecog_v2.pkl")
    arms_classes = arms_lstm.label_encoder.classes_

    ptime = 0
    while True and not stop_e.is_set():
        ctime = time.time()
        hertz = 1 / (ctime - ptime)
        ptime = time.time()

        frame_folder = data_buffer[:]

        if len(frame_folder) == 15:
            start_time = frame_folder[0][0]
            for row_i in range(0, len(frame_folder)):
                frame_folder[row_i][0] -= start_time

            frame_folder = np.array(frame_folder)
            frame_folder = np.expand_dims(frame_folder, axis=0)
            action_class, probabilities = arms_lstm.predict(frame_folder)
            action_class = str(action_class[0])
            probabilities = probabilities[0].tolist()
            results_buffer[:] = [[action_class, list(arms_classes).index(action_class)], probabilities, hertz]

        time.sleep(delay_LSTM)

def signal_uno(commands, stop_e):
    UNO_COM = 'COM9'
    BAUD = 9600

    while not stop_e.is_set():
        try:
            uno = serial.Serial(UNO_COM, BAUD, timeout=1)
            print("Arduino Uno: Connection Success!", flush=True)
            break
        except serial.SerialException as e:
            print(traceback.format_exc(), flush=True)
            print("Arduino Uno: Ensure the bluetooth connection is on.", flush=True)
        except Exception as e:
            print(traceback.format_exc(), flush=True)
            uno.close()

    try:
        uno.write(b"Connection established.\n")
        while not stop_e.is_set():
            if uno.in_waiting > 0:
                data = uno.readline().decode().strip()
                # print(f"Received: {data}", flush=True)

                commands.value = data
        
        uno.close()
    except KeyboardInterrupt:
        uno.close()

def return_most_common_elem(a_list):
    elem_dict = defaultdict(int)
    elem_ind = defaultdict(int)
    for elem in a_list:
        elem_dict[elem] += 1
        elem_ind[elem] += 1
    
    highest_occurrences = max(elem_dict.values())
    curr_largest_avg_index = 0
    most_common_elem = ""
    for elem in elem_ind:
        if elem_dict[elem] == highest_occurrences:
            if (elem_ind[elem] / elem_dict[elem]) > curr_largest_avg_index:
                curr_largest_avg_index = elem_ind[elem] / elem_dict[elem]
                most_common_elem = elem
            elif elem_ind[elem] / elem_dict[elem] == curr_largest_avg_index:
                prev_last_ind = list(a_list)[::-1].index(most_common_elem)
                curr_last_ind = list(a_list)[::-1].index(elem)
                if curr_last_ind < prev_last_ind:
                    most_common_elem = elem

    return most_common_elem

def command_leo(legs_results_buffer, arms_results_buffer, commands, stop_e):
    LEONARDO_COM = "COM7"
    BAUD = 115200

    prev_commands = ""

    # establishing connection
    while not stop_e.is_set():
        try:
            leo = serial.Serial(LEONARDO_COM, BAUD, timeout=1)
            print("Arduino Leonardo: Connection Success!", flush=True)
            break
        except serial.SerialException as e:
            print(traceback.format_exc(), flush=True)
            print("Arduino Leonardo: Ensure the bluetooth connection is on.", flush=True)
        except Exception as e:
            print(traceback.format_exc(), flush=True)


    past_leg_results = deque(maxlen=10) # sliding window of 10 actions, most common action will be the one that is sent to leonardo 
    past_arm_results = deque(maxlen=10)

    while not stop_e.is_set():
        legs_result = ""
        arms_result = ""

        if len(legs_results_buffer) > 0:
            legs_result, legs_index = legs_results_buffer[0]
            legs_probabilities = legs_results_buffer[1]

        if len(arms_results_buffer) > 0:
            arms_result, arms_index = arms_results_buffer[0]
            arms_probabiltiies = arms_results_buffer[1]

        past_leg_results.append(legs_result)
        past_arm_results.append(arms_result)

        normalized_legs_result = return_most_common_elem(past_leg_results)
        normalized_arms_result = return_most_common_elem(past_arm_results)
        msg = normalized_legs_result + "," + normalized_arms_result
        try:
            joystick = commands.value.split(",")
            print(joystick, flush=True)
            if len(joystick) != 3:
                joystick = [0, 0, 1]
            # message format: joystickX, joystickY, joystickPress, legsAction, armsAction, active
            msg = f"{joystick[0]}, {joystick[1]}, {joystick[2]},{normalized_legs_result},{normalized_arms_result},1"
            if msg != prev_commands:
                msg += "\n"
                leo.write(msg.encode(encoding="utf-8"))
                prev_commands = msg
        except:
            print(traceback.format_exc())
            continue

        print(f"Forwarded to Leonardo: {msg}", flush=True)
        time.sleep(delay_LSTM)

    leo.close()


if __name__ == "__main__":
    with Manager() as manager:
        legs_shared_buffer = manager.list()
        arms_shared_buffer = manager.list()
        legs_results_shared_buffer = manager.list()
        arms_results_shared_buffer = manager.list()

        uno_commands = manager.Value('s', "")

        stop_event = manager.Event()

        proc_capture = multip.Process(target=capture_video, args=(legs_shared_buffer, arms_shared_buffer, legs_results_shared_buffer, arms_results_shared_buffer, stop_event))
        proc_lstm_legs = multip.Process(target=legs_lstm, args=(legs_shared_buffer, legs_results_shared_buffer, stop_event))
        proc_lstm_arms = multip.Process(target=arms_lstm, args=(arms_shared_buffer, arms_results_shared_buffer, stop_event))
        proc_uno = multip.Process(target=signal_uno, args=(uno_commands, stop_event))
        proc_leo = multip.Process(target=command_leo, args=(legs_results_shared_buffer, arms_results_shared_buffer, uno_commands, stop_event))

        proc_capture.start()
        proc_lstm_legs.start()
        proc_lstm_arms.start()
        proc_uno.start()
        proc_leo.start()

        stop_event.wait()

        proc_capture.terminate()
        proc_lstm_legs.terminate()
        proc_lstm_arms.terminate()
        proc_uno.terminate()
        proc_leo.terminate()
        # proc_capture.join()
        # proc_lstm_legs.join()
        # proc_lstm_arms.join()
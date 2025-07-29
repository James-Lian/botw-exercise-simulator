import textwrap

from collections import deque
import winsound
import threading

beep_queue = threading.Event()

def beep_listener():
    while True:
        beep_queue.wait()
        winsound.Beep(1000, 300)
        beep_queue.clear()

def beep():
    beep_queue.set()

beep_thread = threading.Thread(target=beep_listener, daemon=True)
beep_thread.start()


# code to gather computer vision + pose data to train machine learning model
import traceback
import time

import math
import serial

import pandas as pd

### receive inputs from IR remote to control play/pause ###
ir_connected = False
ir_port = None
ir_queue = deque(maxlen=300)
ir_commands = deque()

def listen_ir(ser):
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line:
                curr_time = time.time()
                if len(ir_queue) == 0 or curr_time - ir_queue[-1]["time"] > 0.2:
                    ir_commands.append(line)
                    print(line)
                ir_queue.append({"code": line, "time": curr_time})
            else:
                ir_queue.append(0)

def establish_connection():
    while True:
        global ir_port, ir_connected
        if not ir_connected:
            try:
                ir_port = serial.Serial('COM3', 115200)
                ir_connected = True
            except:
                print("IR Port connection failed. Ensure the Arduino is plugged in.")
                print(traceback.format_exc())
                ir_connected = False
                time.sleep(1)
        else:
            listen_ir(ir_port)

ir_thread = threading.Thread(target=establish_connection, daemon=True)
ir_thread.start()


### Finite state machine ###
from enum import Enum

class State(Enum):
    READY = 0
    START_SLEEP = 1
    SLEEPING = 2
    START_RECORDING = 3
    RECORDING = 4
    END_RECORDING = 5
    RESTART_RECORDING = 6
    PAUSE = 7
    BACK = 8
    SKIP = 9

state = State.READY

### ACTION VARIABLES ###
# three separate LSTMs:
## 1 LSTM for arms
## 1 LSTM for legs
## (1 extra for both arms?)
## (RH/LH pose recognizer?)

# walking backwards will be implemented in physical controls as down d-pad
# back-dodge is jumping while shielding

# contains possible combinations
leg_actions = {
    "standing still": [3, 5],
    #shielding while walking is same as running
    "walking": [3, 5],
    "running": [3, 5],
    "jumping": [1],
    "crouching": [3, 5],
    "crouch-walk": [3, 5],
    "dodge-left": [1],
    "dodge-right": [1],
    "strafe-left": [3, 5],
    "strafe-right": [3, 5]
}

# contains durations
arm_actions = {
    # complementary movements for leg movement
    "standing still": [3],
    "walking": [3],
    "running": [3],
    "jumping": [1],
    "crouching": [3],
    "crouch-walk": [3], 

    # actual arm gestures
    "paragliding": [3],
    "paragliding-left": [3],
    "paragliding-right": [3],
    "shielding/lock-on": [3],
    "parry": [1],
    "whistling": [3],  #whisling + running/walking = whistle-sprint
    "swing sword (1h) v1.": [1],
    "swing sword (1h) v2.": [1],
    "special attack (1h)": [3],
    "special attack (1h) release": [1],
    "bow charge": [3],
    "bow release": [1],
    "throw weapon charge": [3],
    "throw weapon release": [1],
    "interact": [3],
    "both arms up": [3],
}

# these actions should be performed together for best training data
action_pairs = {
    "shielding/lock-on": "parry", 
    "swing sword (1h) v1.": "swing sword (1h) v2.", 
    "swing sword (2h) v1.": "swing sword (2h) v2.", 
    "bow charge": "bow release", 
    "throw weapon charge": "throw weapon release",
    "special attack (1h)": "special attack (1h) release"
}

actions = arm_actions.copy()

countdown = 5
rtime = 0 # time value used to measure timings, countdowns, etc.
curr_action = 0
order_of_actions = [] # randomly generated list with multiple actions
timer_for_curr_action = 0
time_limit_for_action = 0

print("Generating action list... ")
import random
for i in range(0, 10):
    all_actions_combos = []
    for action in actions:
        all_actions_combos.append([action, actions[action]])

    randomized_order = []
    while len(all_actions_combos) != 0:
        # index 0 is leg action, index 1 is arm action, index 2 is time choices
        action = random.choice(all_actions_combos)
        if action[0] not in action_pairs.values(): 
            # if arm action is a paired action
            if action[0] in action_pairs: 
                second_action = action_pairs[action[0]]
                randomized_order.append(action)
                randomized_order.append([second_action, actions[second_action]])
                all_actions_combos.remove(action)
                all_actions_combos.remove([second_action, actions[second_action]])
            else:
                randomized_order.append(action)
                all_actions_combos.remove(action)
    order_of_actions.extend(randomized_order)

data = [] # array of dictionaries
curr_recording_data = [] # timestamps and joint positions for current action being recorded

# all joints
all_joints = {}

def return_landmark(landmark):
    return [landmark.x, landmark.y, landmark.z]

def process_landmarks(no, action, timestamp):
    if action[0] in leg_actions:
        action_type = "legs"
    else:
        action_type = "arms"
    snapshot = {
        "action_no": no,
        "action": action[0],
        "action_type": action_type,
        "timestamp": timestamp,
    }
    for joint in all_joints:
        snapshot[joint + "_x"] = all_joints[joint][0]
        snapshot[joint + "_y"] = all_joints[joint][1]
        snapshot[joint + "_z"] = all_joints[joint][2]
    
    return snapshot

### VIDEO VARIABLES ###
import cv2
FONT_BOLD = 2
FONT_COLOUR = (0, 0, 0)
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
model = mp_pose.Pose(model_complexity=1)

cap = cv2.VideoCapture(0)
ptime = 0

while cap.isOpened():
    ### DIAGNOSTICS ###
    # print(state)
    # print(ir_commands)
    # print(curr_action)

    if curr_action == len(order_of_actions):
        break

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

    try:
        landmarks = model_results.pose_world_landmarks.landmark
    except:
        continue

    # face
    all_joints["nose"] = return_landmark(landmarks[mp_pose.PoseLandmark.NOSE.value])
    all_joints["l_ear"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value])
    all_joints["r_ear"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    all_joints["l_eye_inner"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value])
    all_joints["r_eye_inner"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value])
    all_joints["l_eye"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_EYE.value])
    all_joints["r_eye"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value])
    all_joints["l_eye_outer"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value])
    all_joints["r_eye_outer"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value])
    all_joints["l_mouth"] = return_landmark(landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value])
    all_joints["r_mouth"] = return_landmark(landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value])

    # legs
    all_joints["l_hip"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    all_joints["r_hip"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    all_joints["l_knee"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    all_joints["r_knee"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    all_joints["l_ankle"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    all_joints["r_ankle"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    all_joints["l_heel"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value])
    all_joints["r_heel"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value])
    all_joints["l_toe"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])
    all_joints["r_toe"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])

    # arms
    all_joints["l_shoulder"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    all_joints["r_shoulder"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    all_joints["l_elbow"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
    all_joints["r_elbow"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    all_joints["l_wrist"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    all_joints["r_wrist"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    all_joints["l_thumb"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value])
    all_joints["r_thumb"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value])
    all_joints["l_index"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value])
    all_joints["r_index"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value])
    all_joints["l_pinky"] = return_landmark(landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value])
    all_joints["r_pinky"] = return_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value])

    if model_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            model_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    image = cv2.flip(image, 1)
    
    ## listening for IR / State Machine logic
    remote_input = 0
    if len(ir_commands) != 0:
        remote_input = int(ir_commands.popleft())

    ## button inputs
    if remote_input == 45: # start/end (power button)
        if state == State.READY:
            state = State.START_SLEEP
        elif state == State.RECORDING or state == State.SLEEPING or state == State.PAUSE:
            if curr_recording_data != []:
                data.extend(curr_recording_data)
            break
    elif remote_input == 46: # start/finish current motion (vol +)
        if state == State.SLEEPING:
            state = State.START_RECORDING
    elif remote_input == 47: # restart current motion (func/stop)
        state = State.RESTART_RECORDING
    elif remote_input == 43: # skip to next motion (skip)
        state = State.SKIP
    elif remote_input == 44: # redo previous motion (back)
        state = State.BACK
    elif remote_input == 40: # PAUSE/Start again! Take a break
        if state == State.SLEEPING:
            state = State.PAUSE
        elif state == State.PAUSE:
            state = State.START_SLEEP
    
    # timer for motion
    if state == State.RECORDING:
        if ctime - timer_for_curr_action > time_limit_for_action:
            state = State.END_RECORDING

    ## State Machine
    if state == State.SLEEPING: # draw countdown
        countdown_elapsed = ctime - rtime
        if countdown_elapsed >= countdown:
            state = State.START_RECORDING
        
        # overlay + countdown text
        overlay = image.copy()
        gray_color = (220, 220, 220)
        opacity = 0.8
        cv2.rectangle(overlay, (0, 0), (w, h), gray_color, -1)

        image = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

        txt = str(math.ceil(countdown - countdown_elapsed))
        scale = 1.8
        (txt_w, txt_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, FONT_BOLD)
        cv2.putText(image, txt, (int((w-txt_w)/2), int((h-txt_h)/2)), cv2.FONT_HERSHEY_SIMPLEX, scale, FONT_COLOUR, FONT_BOLD)
    elif state == State.START_SLEEP: # start countdown
        beep()
        rtime = time.time()
        state = State.SLEEPING
    elif state == State.START_RECORDING:
        beep()
        rtime = time.time()
        if len(order_of_actions[curr_action][1]) > 1:
            time_limit_for_action = random.randint(order_of_actions[curr_action][1][0], order_of_actions[curr_action][1][1])
        else:
            time_limit_for_action = order_of_actions[curr_action][1][0]
        
        timer_for_curr_action = time.time()
        state = State.RECORDING
    elif state == State.END_RECORDING:
        beep()
        data.extend(curr_recording_data)
        curr_recording_data = []
        curr_action += 1
        state = State.START_SLEEP
    elif state == State.RESTART_RECORDING:
        curr_recording_data = []
        state = State.START_SLEEP
    elif state == State.RECORDING: # recording!!
        curr_recording_data.append(process_landmarks(curr_action, order_of_actions[curr_action], ctime-rtime))
    elif state == State.PAUSE:
        curr_recording_data = []

        # overlay + countdown text
        overlay = image.copy()
        gray_color = (220, 220, 220)
        opacity = 0.8
        cv2.rectangle(overlay, (0, 0), (w, h), gray_color, -1)

        image = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

        txt = "PAUSE"
        scale = 1.5
        (txt_w, txt_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, FONT_BOLD)
        cv2.putText(image, txt, (int((w-txt_w)/2), int((h-txt_h)/2)), cv2.FONT_HERSHEY_SIMPLEX, scale, FONT_COLOUR, FONT_BOLD)
    elif state == State.SKIP:
        beep()
        curr_recording_data = []
        curr_action += 1
        state = State.START_SLEEP
    elif state == State.BACK:
        beep()
        curr_recording_data = []
        curr_action -= 1
        data = [entry for entry in data if entry["action_no"] != curr_action]
        state = State.START_SLEEP

    # display FPS
    cv2.putText(image, "FPS: " + str(int(fps)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_COLOUR, FONT_BOLD)
    # display state
    cv2.putText(image, str(state), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_COLOUR, FONT_BOLD)
    
    # display timer for current action
    if state == State.RECORDING:
        (txt_w, txt_h), _ = cv2.getTextSize(str(time_limit_for_action - math.floor(ctime - timer_for_curr_action)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, FONT_BOLD)
        cv2.putText(image, str(time_limit_for_action - math.floor(ctime - timer_for_curr_action)), (w - 30 - int(txt_w), 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, FONT_COLOUR, FONT_BOLD)
    # action counter
    cv2.putText(image, str(curr_action + 1) + "/" + str(len(order_of_actions)), (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, FONT_COLOUR, FONT_BOLD)

    # display current action
    if state != State.READY and not curr_action >= len(order_of_actions):
        txt = order_of_actions[curr_action][0]
    else:
        txt = "READY"
    (txt_w, txt_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.8, FONT_BOLD)
    cv2.putText(image, txt, (int(w / 2) - int(txt_w/2), h-60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 8)
    cv2.putText(image, txt, (int(w / 2) - int(txt_w/2), h-60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, FONT_COLOUR, 3)

    cv2.imshow("feed", image)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


df = pd.DataFrame(data)
filename = input("Name your file: ")
df.to_csv(filename, index=False)
print("Data written to", filename)
# Zelda: BOTW-Exercise-Simulator (Python / C++ / Arduino) - WIP
Ever wanted to play video games *and* exercise? What's that? Just Dance? Wii Sports? Never heard of 'em.

![GIF of initial test of program](https://github.com/James-Lian/botw-exercise-simulator/blob/main/doc_assets/initial-test.gif)

In this project, I'll detail how I made my own exercise simulator to play Zelda: Breath of the Wild and simulate exactly how taxing saving Hyrule really is. You'll be able to experience both the frustration of forgetting the controls and the exhaustion of swinging at nothing in the air. Link seriously  needs a break.

## 🚧 Status
Currently in development! Results may be unpredictable and unreliable. Features may be added or changed.

## 🖼️ Gallery
#### Showcasing the Action Recognition LSTMS
![GIF of gathering training data for LSTMS](https://github.com/James-Lian/botw-exercise-simulator/blob/main/doc_assets/ActionRecognition.gif)

#### Behind the camera controls - a joystick mounted to an Arduino Uno
![GIF of showcasing camera controls](https://github.com/James-Lian/botw-exercise-simulator/blob/main/doc_assets/behind-the-camera-controls.gif)

## 🧠 How it works
- 📸 Webcam footage is retrieved using OpenCV
- 🏃‍➡️ 3D joint positions and pose landmarks are retrieved from footage using Mediapipe
- 🏭 Joint positions are processed through two [custom-trained LSTMs](https://github.com/James-Lian/action-recognizer): one for recognizing leg movements (e.g. running), and another for recognizing arm movements (e.g. swinging sword)
- 🕹️ An Arduino Uno with an HC-05 bluetooth module mimics the right stick of a Pro controller to turn the camera (unfortunately, as Mediapipe 3D joint positions have their origin centered between the hips, there is no obvious indication or turning or alternative movement, so to control the camera I had to use an Arduino).
  - The joystick data is sent to the PC and forwarded to the Leonardo
- 🎮 An Arduino Leonardo with an HC-05 bluetooth module is plugged into the switch to emulate controller input
  - Once the action recognition is performed, the results are sent to the Leonardo to perform specific controller inputs
- 🤖 Separate processes allow OpenCV, Mediapipe, my LSTMs, and the code to control the Uno and Leonardo all at the same time. The PC acts as the central hub

## 📚 Acknowledgements
- 🐍 Python libraries used include OpenCV, Mediapipe, Tensorflow, Multiprocessing, Pyserial, Pandas, and more
- ▶️ Thomas Hansknecht video on [Automating MarioKart with an Arduino Leonardo](https://www.youtube.com/watch?v=a1I5drxKfBY), who showed me that this project was possible
  - His github repository can be found [here](https://github.com/tfh0007/MarioKartScript), which contains instructions
  - Some setup is required to use the Leonardo to emulate a Switch Controller. See [here](https://github.com/James-Lian/botw-exercise-simulator/tree/main/leonardo-switchcontroller)
- 📑 lefmarna's [NintendoSwitchControlLibrary](https://github.com/lefmarna/NintendoSwitchControlLibrary) which allowed the Arduino Leonardo to act as a switch controller
  - Subsequently credit should also be given to celclow's [SwitchControlLibrary](https://github.com/celclow/SwitchControlLibrary) which inspired lefmarna's

## 📈 Timeline
- Early June 2025: Start project and brainstorming
- Mid June 2025: Training custom LSTMs using gathered joint data (see [this repository](https://github.com/James-Lian/action-recognizer))
- Early July 2025: Building Arduino Uno and Arduino Leonardo circuits
- July 28th, 2025: First working version
- Currently: Improving the action recognition accuracy, decreasing latency between commands, adding more actions and functionality, and improving overall performance

## 🕹️ How to run
⚠️ As of the time of writing (July 2025), Mediapipe only works with Python version 3.12 or earlier versions.

## ⌨️ Command List
Video coming soon (?)

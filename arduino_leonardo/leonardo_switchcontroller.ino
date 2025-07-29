#include <Arduino.h>
#include <NintendoSwitchControlLibrary.h>

int buzzerPin = 12;

String message = "";

void setup() {
  Serial.begin(115200);      // USB serial for debugging
  Serial1.begin(115200);     // HC-05 communication

  pinMode(buzzerPin, OUTPUT);
}

String prevLegsAction = "";
String prevArmsAction = "";

// Switch Controls
const int SHORTPRESS = 25;
const int MEDIUMPRESS = 100;
const int REPEATEDPRESSLATENCY = 50;
struct SwitchButtons {
  uint16_t name;
  unsigned long start; // start time
  unsigned long duration; // duration of buttonPress
};
SwitchButtons buttonPresses[] = {
  {Button::Y, 0, 0},
  {Button::B, 0, 0},
  {Button::A, 0, 0},
  {Button::X, 0, 0},
  {Button::L, 0, 0},
  {Button::R, 0, 0},
  {Button::ZL, 0, 0},
  {Button::ZR, 0, 0},
  {Button::MINUS, 0, 0},
  {Button::PLUS, 0, 0},
  {Button::LCLICK, 0, 0},
  {Button::RCLICK, 0, 0},
  {Button::HOME, 0, 0},
  {Button::CAPTURE, 0, 0}
};
SwitchButtons buttonPressQueue[] = {
  {Button::Y, 0, 0},
  {Button::B, 0, 0},
  {Button::A, 0, 0},
  {Button::X, 0, 0},
  {Button::L, 0, 0},
  {Button::R, 0, 0},
  {Button::ZL, 0, 0},
  {Button::ZR, 0, 0},
  {Button::MINUS, 0, 0},
  {Button::PLUS, 0, 0},
  {Button::LCLICK, 0, 0},
  {Button::RCLICK, 0, 0},
  {Button::HOME, 0, 0},
  {Button::CAPTURE, 0, 0}
};
// or D-pad
struct SwitchHat {
  uint8_t name;
  unsigned long start;
  unsigned long duration;
};
SwitchHat hatPresses[] = {
  {Hat::UP, 0, 0},
  {Hat::LEFT, 0, 0},
  {Hat::RIGHT, 0, 0},
  {Hat::DOWN, 0, 0}
};

bool isCrouching = false;
bool isScoping = false;
bool isGuarding = false;


// leg_actions = {
//     "standing still": [3, 5],
//     "walking": [3, 5],
//     "running": [3, 5],
//     "jumping": [1],
//     "crouching": [3, 5],
//     "crouch-walk": [3, 5],
//     "dodge-left": [1],
//     "dodge-right": [1],
//     "strafe-left": [3, 5],
//     "strafe-right": [3, 5]
// }

// arm_actions = {
//     "standing still": [3],
//     "walking": [3],
//     "running": [3],
//     "jumping": [1],
//     "crouching": [3],
//     "crouch-walk": [3], 

//     "paragliding": [3],
//     "paragliding-left": [3],
//     "paragliding-right": [3],
//     "shielding/lock-on": [3],
//     "parry": [1],
//     "whistling": [3],
//     "swing sword (1h) v1.": [1],
//     "swing sword (1h) v2.": [1],
//     "special attack (1h)": [3],
//     "special attack (1h) release": [1],
//     "bow charge": [3],
//     "bow release": [1],
//     "throw weapon charge": [3],
//     "throw weapon release": [1],
//     "interact": [3],
//     "both arms up": [3],
// }

void pressButton(unsigned int button, unsigned long stime, unsigned long duration) {
  SwitchControlLibrary().pressButton(button);
  for (auto& iButton : buttonPresses) {
    iButton.start = stime;
    iButton.duration = duration;
  }
}

void pressHat(unsigned int hat, unsigned long stime, unsigned long duration) {
  SwitchControlLibrary().pressHatButton(hat);
  for (auto& iHat : hatPresses) {
    iHat.start = stime;
    iHat.duration = duration;
  }
}

void crouch(bool state, unsigned long time) {
  if (isCrouching != state) {
    isCrouching = state;
    pressButton(Button::LCLICK, time, SHORTPRESS);
  }
}


void loop() {
  unsigned long cTime = millis();

  while (Serial1.available()) {
    cTime = millis();

    char c = Serial1.read();
    message += c;

    if (c == '\n') {
      Serial.print("Arduino got: ");
      Serial.print(message);

      String tokens[8];
      int tokenIndex = 0;
      int lastIndex = 0;
      for (int i = 0; i < message.length(); i++) {
        if (message[i] == ',') {
          tokens[tokenIndex++] = message.substring(lastIndex, i);
          lastIndex = i + 1;
        }
      }
      // Add last token
      tokens[tokenIndex++] = message.substring(lastIndex);

      // Trim spaces
      for (int i = 0; i < tokenIndex; i++) {
        tokens[i].trim();
      }

      // Assign values by index
      unsigned int joystickX = tokens[0].toInt();
      joystickX = map(joystickX, 0, 1023, 0, 255);
      unsigned int joystickY = tokens[1].toInt();
      joystickY = map(joystickY, 0, 1023, 0, 255);
      unsigned int joystickButton = tokens[2].toInt();
      String legsAction = tokens[3];
      String armsAction = tokens[4];
      String active = tokens[5];

      Serial.print(legsAction);
      Serial.print(", ");
      Serial.print(prevLegsAction);
      Serial.print(" | ");
      Serial.print(armsAction);
      Serial.print(", ");
      Serial.println(prevArmsAction);

      SwitchControlLibrary().moveRightStick(joystickX, joystickY);

      // leg actions
      if (legsAction != prevLegsAction) {
        Serial.println(1);
        crouch(false, cTime);
        if (legsAction == "standing still" || legsAction == "walking" || armsAction == "both arms up") {
          Serial.println(11);
          SwitchControlLibrary().moveLeftStick(Stick::NEUTRAL, Stick::NEUTRAL);
        }
        else if (legsAction == "running") {
          Serial.println(12);
          SwitchControlLibrary().moveLeftStick(Stick::NEUTRAL, Stick::MIN);
        }
        else if (legsAction == "jumping") {
          Serial.println(13);
          pressButton(Button::X, cTime, SHORTPRESS);
          // backwards dodge
          if (armsAction == "shielding/lock-on") {
            SwitchControlLibrary().moveLeftStick(Stick::NEUTRAL, Stick::MAX);
          }
        }
        else if (legsAction == "crouching") {
          Serial.println(14);
          crouch(true, cTime);
          if (armsAction == "crouch-walk") {
            SwitchControlLibrary().moveLeftStick(Stick::NEUTRAL, Stick::MIN);
          }
        }
        else if (legsAction == "crouch-walk") {
          Serial.println(15);
          crouch(true, cTime);
          SwitchControlLibrary().moveLeftStick(Stick::NEUTRAL, Stick::MIN);
        }
        else if (legsAction == "dodge-left") {
          Serial.println(16);
          if (armsAction == "shielding/lock-on" || prevArmsAction == "shielding/lock-on") {
            pressButton(Button::X, cTime, SHORTPRESS);
            SwitchControlLibrary().moveLeftStick(Stick::MIN, Stick::NEUTRAL);
          }
        }
        else if (legsAction == "dodge-right") {
          Serial.println(17);
          if (armsAction == "shielding/lock-on" || prevArmsAction == "shielding/lock-on") {
            pressButton(Button::X, cTime, SHORTPRESS);
            SwitchControlLibrary().moveLeftStick(Stick::MAX, Stick::NEUTRAL);
          }
        }
        else if (legsAction == "strafe-left") {
          Serial.println(18);
          SwitchControlLibrary().moveLeftStick(Stick::MIN, Stick::NEUTRAL);
        }
        else if (legsAction == "strafe-right") {
          Serial.println(19);
          SwitchControlLibrary().moveLeftStick(Stick::MAX, Stick::NEUTRAL);
        }

        prevLegsAction = legsAction;
      }

      // arm actions
      if (armsAction != prevArmsAction) {
        Serial.println(2);
        isGuarding = false;
        if (prevArmsAction == "special attack (1h)" && (
            armsAction != "special attack (1h) release" && 
            armsAction != "swing sword (1h) v1." && 
            armsAction != "swing sword (1h) v2."
          )) {
            Serial.println(21);
            SwitchControlLibrary().releaseButton(Button::Y);
            pressButton(Button::B, cTime, SHORTPRESS);
        }
        else if (prevArmsAction == "throw weapon charge" && (
          armsAction != "throw weapon release" &&
          armsAction != "swing sword (1h) v1." && 
          armsAction != "swing sword (1h) v2."
        )) {
          Serial.println(22);
          SwitchControlLibrary().releaseButton(Button::R);
          pressButton(Button::B, cTime, SHORTPRESS);
        }
        else if (prevArmsAction == "bow charge" && armsAction != "bow release") {
          Serial.println(23);
          SwitchControlLibrary().releaseButton(Button::R);
          pressButton(Button::B, cTime, SHORTPRESS);
        }

        if (armsAction == "both arms up") {
          isGuarding = false;
          pressButton(Button::B, cTime, SHORTPRESS);
        }
        else if (armsAction == "walking") {
          Serial.println(24);
          isGuarding = true;
        }
        // whistle-spring
        else if ((legsAction == "running") && armsAction == "whistling") {
          Serial.println(25);
          SwitchControlLibrary().moveLeftStick(Stick::NEUTRAL, Stick::MIN);
          // something like this will need a function
        }
        else if (armsAction == "whistling") {
          Serial.println(26);
          pressHat(Hat::DOWN, cTime, SHORTPRESS);
        }
        else if (armsAction == "paragliding" || armsAction == "jumping") {
          Serial.println(27);
          if (prevLegsAction == "jumping" || legsAction == "jumping") {
            SwitchControlLibrary().pressButton(Button::X);
          }
          // if jump button wasn't yet pressed
          else {
            for (auto& entry : buttonPresses) {
              if (entry.name == Button::X) {
                entry.start = cTime + SHORTPRESS + REPEATEDPRESSLATENCY;
                entry.duration = SHORTPRESS;
              }
            }
          }
        }
        else if (armsAction == "paragliding-left") {
          Serial.println(28);
          SwitchControlLibrary().pressButton(Button::X);
          SwitchControlLibrary().moveLeftStick(Stick::MIN, Stick::NEUTRAL);
        }
        else if (armsAction == "paragliding-right") {
          Serial.println(29);
          SwitchControlLibrary().pressButton(Button::X);
          SwitchControlLibrary().moveLeftStick(Stick::MIN, Stick::NEUTRAL);
        }
        else if (armsAction == "shielding/lock-on") {
          Serial.println(210);
          SwitchControlLibrary().pressButton(Button::ZL);
          isGuarding = true;
        }
        else if (armsAction == "parry") {
          Serial.println(211);
          if (prevArmsAction == "shielding/lock-on") {
            pressButton(Button::A, cTime, SHORTPRESS);
          }
          isGuarding = true;
        }
        else if (armsAction == "swing sword (1h) v1." || armsAction == "swing sword (1h) v2.") {
          Serial.println(212);
          if (prevArmsAction == "throw weapon charge") {
            SwitchControlLibrary().releaseButton(Button::R);
          }
          else {
            pressButton(Button::Y, cTime, SHORTPRESS);
          }
        }
        else if (armsAction == "special attack (1h)") {
          Serial.println(213);
          SwitchControlLibrary().pressButton(Button::Y);
        }
        else if (armsAction == "special attack (1h) release") {
          Serial.println(214);
          if (prevArmsAction == "special attack (1h)") {
            SwitchControlLibrary().releaseButton(Button::Y);
          }
        }
        else if (armsAction == "throw weapon charge") {
          Serial.println(215);
          SwitchControlLibrary().pressButton(Button::R);
        }
        else if (armsAction == "throw weapon release") {
          Serial.println(216);
          if (prevArmsAction == "throw weapon charge") {
            SwitchControlLibrary().releaseButton(Button::R);
          }
          else {
            pressButton(Button::Y, cTime, SHORTPRESS);
          }
        }
        else if (armsAction == "bow charge") {
          Serial.println(217);
          SwitchControlLibrary().pressButton(Button::ZR);
        }
        else if (armsAction == "bow release") {
          Serial.println(218);
          if (prevArmsAction == "bow charge") {
            SwitchControlLibrary().releaseButton(Button::ZR);
          }
          else {
            pressButton(Button::ZR, cTime, MEDIUMPRESS);
          }
        }
        else if (armsAction == "interact") {
          Serial.println(219);
          pressButton(Button::A, cTime, SHORTPRESS);
        }

        // cancelling whistle for whistle-sprint
        if ((prevLegsAction == "running" || prevLegsAction == "running") && prevArmsAction == "whistling") {
          Serial.println(220);
          pressButton(Button::B, cTime, SHORTPRESS);
        }

        prevArmsAction = armsAction;
      }

      
      Serial.println(3);

      if (isGuarding == false) {
        SwitchControlLibrary().releaseButton(Button::ZL);
      }

      // independent leg actions

      // independent arm actions
      Serial.println(31);
      // schedule button presses
      for (auto& entry : buttonPressQueue) {
        if (entry.start <= cTime && entry.start != 0) {
          pressButton(entry.name, cTime, entry.duration);
          entry.start = 0;
          entry.duration = 0;
        }
      }

      Serial.println(32);

      // releasing any button presses that have reached their time
      for (auto& entry : buttonPresses) {
        if (entry.start + entry.duration <= cTime && entry.start != 0) {
          SwitchControlLibrary().releaseButton(entry.name);
          entry.start = 0;
          entry.duration = 0;
        }
      }
      Serial.println(33);
      for (auto&entry : hatPresses) {
        if (entry.start + entry.duration <= cTime && entry.start != 0) {
          SwitchControlLibrary().releaseHatButton();
          entry.start = 0;
          entry.duration = 0;
        }
      }

      Serial.println("Done");
      
      message = "";
    }
  }
  SwitchControlLibrary().sendReport();
}
#include <SoftwareSerial.h>

SoftwareSerial BT(4, 5);

int JOYSTICK_X = 0;
int JOYSTICK_Y = 1;
int JOYSTICK_SW = 2;

int xVal;
int yVal;
int buttonState;

void setup() {
  Serial.begin(9600);
  BT.begin(9600);
  
  pinMode(JOYSTICK_SW, INPUT_PULLUP);
}

void loop() {
  xVal = analogRead(JOYSTICK_X);
  yVal = analogRead(JOYSTICK_Y);

  buttonState = digitalRead(JOYSTICK_SW);
  
  Serial.print("X: ");
  Serial.print(xVal);
  Serial.print(" | Y: ");
  Serial.print(yVal);
  Serial.print(" | Button: ");
  Serial.println(buttonState);
  BT.print(xVal);
  BT.print(",");
  BT.print(yVal);
  BT.print(",");
  BT.println(buttonState);
  delay(100);
}

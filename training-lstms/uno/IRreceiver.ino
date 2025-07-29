#include <IRremote.hpp>

const int IR_RECEIVE_PIN = 2;

void setup() {
    Serial.begin(115200);
    IrReceiver.begin(IR_RECEIVE_PIN, ENABLE_LED_FEEDBACK); // Start receiver
}

void loop() {
    if (IrReceiver.decode()) {
        Serial.println(IrReceiver.decodedIRData.command, HEX);

        IrReceiver.resume(); // Prepare to receive the next signal
    }
}
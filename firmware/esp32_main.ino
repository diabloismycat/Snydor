#include <Arduino.h>

void setup() {
  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');

    if (cmd == "OPEN") {
      // servo open
    }
    else if (cmd == "GRASP") {
      // servo grasp
    }
    else if (cmd == "PINCH") {
      // pinch
    }
  }
}

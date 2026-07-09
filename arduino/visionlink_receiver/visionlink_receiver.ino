/*
 * VisionLink Arduino Receiver
 *
 * Upload to an Arduino Uno (or compatible board).
 * Open the Serial Monitor at 9600 baud to see incoming commands.
 *
 * Wiring (optional LED feedback on pin 13 — built-in LED on Uno):
 *   Uses LED_BUILTIN (pin 13)
 */

const int LED_PIN = LED_BUILTIN;
const unsigned long BLINK_MS = 100;

String buffer;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(9600);
  buffer.reserve(32);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (buffer.length() > 0) {
        handleCommand(buffer);
        buffer = "";
      }
    } else {
      buffer += c;
    }
  }
}

void handleCommand(const String &cmd) {
  Serial.print(F("RX: "));
  Serial.println(cmd);

  if (cmd == "FACE_FOUND") {
    blink(1);
  } else if (cmd == "NO_FACE") {
    digitalWrite(LED_PIN, LOW);
  } else if (cmd == "SMILE") {
    blink(3);
  } else if (cmd == "LEFT" || cmd == "RIGHT" || cmd == "UP" || cmd == "DOWN") {
    blink(2);
  } else {
    // MOUTH_OPEN, EYES_CLOSED, etc. — short blink
    blink(1);
  }
}

void blink(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(BLINK_MS);
    digitalWrite(LED_PIN, LOW);
    delay(BLINK_MS);
  }
}

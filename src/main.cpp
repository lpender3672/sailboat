// RP2040 (Raspberry Pi Pico) + CRSF receiver + motor driver
#include <Arduino.h>
#include <CRSFforArduino.hpp>

// Serial port for CRSF on Pico: Serial1 defaults to GP4 (TX) / GP5 (RX)
// Wire RX (from receiver) -> GP5, TX (to receiver) -> GP4
CRSFforArduino crsf(&Serial1);

// BTS7960 pins on Pico (PWM capable pairs suggested)
// Use PWM-capable GPIOs for RPWM/LPWM (e.g., GP2, GP3)
const uint8_t PIN_R_EN = 10;  // GP10
const uint8_t PIN_L_EN = 11;  // GP11
const uint8_t PIN_RPWM = 2;   // GP2 (PWM)
const uint8_t PIN_LPWM = 3;   // GP3 (PWM)
const uint8_t PIN_R_IS = 26;  // GP26/ADC0 (optional)
const uint8_t PIN_L_IS = 27;  // GP27/ADC1 (optional)

static inline void enableDriver(bool enable) {
  digitalWrite(PIN_R_EN, enable ? HIGH : LOW);
  digitalWrite(PIN_L_EN, enable ? HIGH : LOW);
}

// speed: -255..255 (negative = reverse)
static inline void setSpeed(int16_t speed) {
  if (speed > 255) speed = 255;
  if (speed < -255) speed = -255;
  if (speed > 0) {
    analogWrite(PIN_RPWM, speed);
    analogWrite(PIN_LPWM, 0);
  } else if (speed < 0) {
    analogWrite(PIN_RPWM, 0);
    analogWrite(PIN_LPWM, -speed);
  } else {
    analogWrite(PIN_RPWM, 0);
    analogWrite(PIN_LPWM, 0);
  }
}

static inline uint16_t readCurrentRaw() {
  return analogRead(PIN_R_IS) + analogRead(PIN_L_IS);
}

// Map CRSF channel (1000-2000us) to -255..255 motor speed
int16_t usToSpeed(int us) {
  if (us < 1000) us = 1000;
  if (us > 2000) us = 2000;
  long val = map(us, 1000, 2000, -255, 255);
  return (int16_t)val;
}

void setup() {
  Serial.begin(115200);
  // Start CRSF on Serial1 (defaults are fine)
  crsf.begin();

  pinMode(PIN_R_EN, OUTPUT);
  pinMode(PIN_L_EN, OUTPUT);
  pinMode(PIN_RPWM, OUTPUT);
  pinMode(PIN_LPWM, OUTPUT);
  pinMode(PIN_R_IS, INPUT);
  pinMode(PIN_L_IS, INPUT);
  enableDriver(true);
  setSpeed(0);
}

void loop() {
  crsf.update();

  // Use CH3 (throttle-like) to drive motor
  uint16_t ch3_us = crsf.rcToUs(crsf.getChannel(3));
  int16_t speed = usToSpeed((int)ch3_us);
  setSpeed(speed);

  // Optional: print debug
  static uint32_t last = 0;
  uint32_t now = millis();
  if (now - last > 250) {
    last = now;
    Serial.print("CH3(us): "); Serial.print(ch3_us);
    Serial.print("  speed: "); Serial.print(speed);
    Serial.print("  Iraw: "); Serial.println(readCurrentRaw());
  }
}
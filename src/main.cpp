// RP2040 (Raspberry Pi Pico) + CRSF receiver + motor driver
#include <Arduino.h>
#include <CRSFforArduino.hpp>
#include <Servo.h>

#include "hardware/uart.h"
#include "hardware/gpio.h"

CRSFforArduino crsf(&Serial1, 1, 0);

// Low-level UART configuration (matches your known-good sequence)
#define UART_ID      uart0
#define BAUD_RATE    420000
#define DATA_BITS    8
#define STOP_BITS    1
#define PARITY       UART_PARITY_NONE
#define UART_TX_PIN  0
#define UART_RX_PIN  1

const uint8_t PIN_R_EN = 10;
const uint8_t PIN_L_EN = 11;
const uint8_t PIN_RPWM = 2;
const uint8_t PIN_LPWM = 3;
const uint8_t PIN_R_IS = 26;  // ADC0
const uint8_t PIN_L_IS = 27;  // ADC1

const uint8_t PIN_SERVO = 4;
Servo servo;

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


int16_t usToSpeed(int us) {
  if (us < 1000) us = 1000;
  if (us > 2000) us = 2000;
  long val = map(us, 1000, 2000, -255, 255);
  return (int16_t)val;
}


static volatile bool g_linkUp = false;


static void onRcChannels(serialReceiverLayer::rcChannels_t *rcChannels);

static void onLinkUp() {
  g_linkUp = true;
  enableDriver(true);
  setSpeed(0);
  Serial.println("CRSF link: UP");
}

static void onLinkDown() {
  g_linkUp = false;
  setSpeed(0);
  enableDriver(false);
  Serial.println("CRSF link: DOWN");
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_R_EN, OUTPUT);
  pinMode(PIN_L_EN, OUTPUT);
  pinMode(PIN_RPWM, OUTPUT);
  pinMode(PIN_LPWM, OUTPUT);
  pinMode(PIN_R_IS, INPUT);
  pinMode(PIN_L_IS, INPUT);
  enableDriver(false);
  setSpeed(0);

  // Servo setup: 1000-2000us range
  servo.attach(PIN_SERVO, 1000, 2000);
  servo.writeMicroseconds(1500);

  // CRSF link callbacks
  crsf.setLinkUpCallback(onLinkUp);
  crsf.setLinkDownCallback(onLinkDown);
  crsf.setRcChannelsCallback(onRcChannels);

  // magic uart setup (anything missing and it breaks)
  uart_init(UART_ID, BAUD_RATE);
  gpio_set_function(UART_TX_PIN, GPIO_FUNC_UART);
  gpio_set_function(UART_RX_PIN, GPIO_FUNC_UART);
  uart_set_hw_flow(UART_ID, false, false);
  uart_set_format(UART_ID, DATA_BITS, STOP_BITS, PARITY);
  uart_set_fifo_enabled(UART_ID, false);

  crsf.begin(BAUD_RATE);
  Serial.println("Waiting for CRSF link...");

  uint32_t t0 = millis();
  while (!g_linkUp && (millis() - t0 < 8000)) {
    crsf.update();
    Serial.print(".");
    delay(50);
  }
  if (!g_linkUp) {
    Serial.println("\nCRSF link not detected (timeout). Motor stays disabled.");
  }
}

void loop() {
  crsf.update();

  if (!g_linkUp) {
    setSpeed(0);
  servo.writeMicroseconds(1500);
    return;
  }
  

  // Use CH3 (throttle-like) to drive motor
  uint16_t ch3_us = crsf.rcToUs(crsf.getChannel(3));
  int16_t speed = usToSpeed((int)ch3_us);
  setSpeed(speed);

  // Channel 4 -> servo (e.g., yaw)
  uint16_t ch4_us = crsf.rcToUs(crsf.getChannel(4));
  servo.writeMicroseconds(ch4_us);


  static uint32_t last = 0;
  uint32_t now = millis();
  if (now - last > 250) {
    last = now;
  Serial.print("CH3(us): "); Serial.print(ch3_us);
    Serial.print("  speed: "); Serial.print(speed);
  Serial.print("  CH4(us): "); Serial.print(ch4_us);
    Serial.print("  Iraw: "); Serial.println(readCurrentRaw());
  }
}

// Called whenever RC channels are received; treat as link present
static void onRcChannels(serialReceiverLayer::rcChannels_t * /*rcChannels*/) {
  if (!g_linkUp) {
    onLinkUp();
  }
}
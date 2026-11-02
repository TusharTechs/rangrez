// Rangrez rig firmware (ESP32, Arduino core). One JSON command per line over USB serial.
//
// Every command is checked against hard limits here, independently of the host:
// a host bug cannot overfill the cup, run a pump dry, or overheat the tape.
//
// Libraries: ArduinoJson (v7), ESP32Servo, AccelStepper.

#include <Arduino.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include <AccelStepper.h>
#include <esp_task_wdt.h>

// ---- pins ---------------------------------------------------------------
const int PUMP_PINS[5] = {16, 17, 18, 19, 21};  // turquoise, fuchsia, yellow, water, drain (MOSFET gates)
const int STIR_PIN = 22;                         // small DC stir motor via MOSFET
const int DRYER_PIN = 23;                        // 12 V blower relay
const int LIGHT_PIN = 25;                        // LED panel MOSFET
const int DIP_SERVO_PIN = 26;                    // lifts the cup up to the tape
const int STEP_PINS[4] = {27, 14, 12, 13};       // 28BYJ-48 via ULN2003
const int ESTOP_PIN = 34;                        // normally-closed button to GND + external 10k pull-up to 3.3 V
                                                 // (GPIO 34 is input-only with no internal pull-up); a pressed
                                                 // button or a broken wire reads HIGH and stops the rig

// ---- limits (also enforced on the host) ---------------------------------
const float MAX_PUMP_ML = 20.0;      // per command
const float MAX_BATH_ML = 25.0;      // liquid in the cup between drains
const float MAX_DIP_S = 300.0;
const float MAX_DRY_S = 180.0;
const float MAX_STIR_S = 30.0;
const float MAX_DRAIN_S = 20.0;
const float MAX_ADVANCE_MM = 150.0;

// ---- calibration (measured by `rangrez-calibrate`; ms per mL per pump) ---
const float MS_PER_ML_COMPILED[5] = {950, 950, 950, 900, 0};
float msPerMl[5] = {950, 950, 950, 900, 0};
const float MAX_CALIBRATION_CHANGE = 0.20;  // a host can fine-tune a pump by ±20%, never triple its output
const int WATCHDOG_S = 8;                    // a hung loop resets the board, which starts with everything off
const int DIP_UP_DEG = 20, DIP_DOWN_DEG = 105;
const float STEPS_PER_MM = 2048.0 / 94.2;  // 30 mm drive roller

Servo dipServo;
AccelStepper tape(AccelStepper::HALF4WIRE, STEP_PINS[0], STEP_PINS[2], STEP_PINS[1], STEP_PINS[3]);
float bathMl = 0;
volatile bool estop = false;

void IRAM_ATTR onEstop() { estop = true; }

void allOff() {
  for (int p : PUMP_PINS) digitalWrite(p, LOW);
  digitalWrite(STIR_PIN, LOW);
  digitalWrite(DRYER_PIN, LOW);
  dipServo.write(DIP_UP_DEG);
}

void reply(const char* cmd, const char* error = nullptr) {
  JsonDocument out;
  if (error) out["error"] = error; else out["done"] = cmd;
  out["bath_ml"] = bathMl;
  serializeJson(out, Serial);
  Serial.println();
}

// Waits in small steps so the e-stop can interrupt anything.
bool waitMs(unsigned long ms) {
  unsigned long until = millis() + ms;
  while (millis() < until) {
    esp_task_wdt_reset();
    if (estop) { allOff(); return false; }
    delay(5);
  }
  return true;
}

// False if the e-stop cut it short.
bool runFor(int pin, float seconds) {
  digitalWrite(pin, HIGH);
  bool ok = waitMs((unsigned long)(seconds * 1000));
  digitalWrite(pin, LOW);
  return ok;
}

void setup() {
  Serial.begin(115200);
  esp_task_wdt_init(WATCHDOG_S, true);  // panic and reboot on a hang
  esp_task_wdt_add(NULL);
  for (int p : PUMP_PINS) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }
  pinMode(STIR_PIN, OUTPUT); pinMode(DRYER_PIN, OUTPUT); pinMode(LIGHT_PIN, OUTPUT);
  pinMode(ESTOP_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(ESTOP_PIN), onEstop, RISING);
  if (digitalRead(ESTOP_PIN) == HIGH) estop = true;  // pressed or unplugged at power-up: stay stopped
  dipServo.attach(DIP_SERVO_PIN);
  tape.setMaxSpeed(600); tape.setAcceleration(400);
  allOff();
}

void loop() {
  esp_task_wdt_reset();
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  JsonDocument doc;
  if (deserializeJson(doc, line)) { reply("?", "bad json"); return; }
  const char* cmd = doc["cmd"] | "";
  if (estop && strcmp(cmd, "reset") != 0 && strcmp(cmd, "ping") != 0) { reply(cmd, "e-stop latched; send reset"); return; }

  if (!strcmp(cmd, "ping")) {
    JsonDocument out; out["fw"] = "rangrez-1"; out["estop"] = estop; out["bath_ml"] = bathMl;
    serializeJson(out, Serial); Serial.println();
  } else if (!strcmp(cmd, "reset")) {
    if (digitalRead(ESTOP_PIN) == HIGH) return reply(cmd, "e-stop still pressed: release it first");
    estop = false; allOff(); reply(cmd);
  } else if (!strcmp(cmd, "pump")) {
    int id = doc["id"] | -1; float ml = doc["ml"] | 0.0;
    if (id < 0 || id > 3) return reply(cmd, "pump id must be 0-3 (use drain for 4)");
    if (ml <= 0 || ml > MAX_PUMP_ML) return reply(cmd, "ml out of range");
    if (bathMl + ml > MAX_BATH_ML) return reply(cmd, "cup would overflow");
    bool ok = runFor(PUMP_PINS[id], ml * msPerMl[id] / 1000.0);
    bathMl += ml;  // worst case, for the overflow check
    ok ? reply(cmd) : reply(cmd, "e-stop during pump");
  } else if (!strcmp(cmd, "drain")) {
    float s = doc["seconds"] | 0.0;
    if (s <= 0 || s > MAX_DRAIN_S) return reply(cmd, "drain seconds out of range");
    if (!runFor(PUMP_PINS[4], s)) return reply(cmd, "e-stop during drain");
    bathMl = 0; reply(cmd);
  } else if (!strcmp(cmd, "stir")) {
    float s = doc["seconds"] | 0.0;
    if (s <= 0 || s > MAX_STIR_S) return reply(cmd, "stir seconds out of range");
    runFor(STIR_PIN, s) ? reply(cmd) : reply(cmd, "e-stop during stir");
  } else if (!strcmp(cmd, "dip")) {
    float s = doc["seconds"] | 0.0;
    if (s < 10 || s > MAX_DIP_S) return reply(cmd, "dip seconds out of range");
    if (bathMl < 10) return reply(cmd, "cup is nearly empty");
    dipServo.write(DIP_DOWN_DEG);
    bool ok = waitMs((unsigned long)(s * 1000));
    dipServo.write(DIP_UP_DEG);
    waitMs(800);
    if (!ok) return reply(cmd, "e-stop during dip");
    reply(cmd);
  } else if (!strcmp(cmd, "advance")) {
    float mm = doc["mm"] | 0.0;
    if (mm <= 0 || mm > MAX_ADVANCE_MM) return reply(cmd, "advance mm out of range");
    tape.move((long)(mm * STEPS_PER_MM));
    while (tape.distanceToGo() != 0) { esp_task_wdt_reset(); if (estop) { allOff(); return reply(cmd, "e-stop"); } tape.run(); }
    tape.disableOutputs();
    reply(cmd);
  } else if (!strcmp(cmd, "dry")) {
    float s = doc["seconds"] | 0.0;
    if (s <= 0 || s > MAX_DRY_S) return reply(cmd, "dry seconds out of range");
    runFor(DRYER_PIN, s) ? reply(cmd) : reply(cmd, "e-stop during dry");
  } else if (!strcmp(cmd, "light")) {
    digitalWrite(LIGHT_PIN, (doc["on"] | false) ? HIGH : LOW); reply(cmd);
  } else if (!strcmp(cmd, "calibrate")) {
    int id = doc["id"] | -1; float ms = doc["ms_per_ml"] | 0.0;
    if (id < 0 || id > 3 || ms < 100 || ms > 5000) return reply(cmd, "bad calibration");
    if (fabs(ms - MS_PER_ML_COMPILED[id]) > MAX_CALIBRATION_CHANGE * MS_PER_ML_COMPILED[id])
      return reply(cmd, "calibration more than 20% from the compiled value: reflash to change it");
    msPerMl[id] = ms; reply(cmd);
  } else {
    reply(cmd, "unknown command");
  }
}

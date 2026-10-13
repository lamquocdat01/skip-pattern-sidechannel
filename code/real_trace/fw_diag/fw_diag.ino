// fw_diag.ino — T22a step A2: I2C scan + 20 INA219 samples on SDA=GPIO8, SCL=GPIO9.
//
// Calibration: Adafruit setCalibration_32V_2A() = CAL 4096 for the 0.1 ohm shunt,
// current LSB 0.1 mA, power LSB 2 mW — the same register scaling as the C2b driver
// (cfs-mcu/main/ina219.cc). Energy work uses the CURRENT register, never POWER.
//
// The whole report repeats every 10 s so the host can attach at any time.
#include <Wire.h>
#include <Adafruit_INA219.h>

static const int PIN_SDA = 8;
static const int PIN_SCL = 9;
static const uint8_t INA_ADDR = 0x40;

Adafruit_INA219 ina(INA_ADDR);

static int read_pin(int pin, bool pullup) {
  pinMode(pin, pullup ? INPUT_PULLUP : INPUT_PULLDOWN);
  delay(20);
  return digitalRead(pin);
}

static uint16_t read_reg(uint8_t reg, bool *ok) {
  Wire.beginTransmission(INA_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) { *ok = false; return 0; }
  if (Wire.requestFrom((int)INA_ADDR, 2) != 2) { *ok = false; return 0; }
  uint16_t v = ((uint16_t)Wire.read() << 8) | Wire.read();
  *ok = true;
  return v;
}

static void report() {
  Serial.println();
  Serial.println("=== T22A_DIAG begin ===");
  Serial.printf("pins SDA=%d SCL=%d addr=0x%02X cal=32V_2A (CAL 4096, I_LSB 0.1 mA, P_LSB 2 mW)\n",
                PIN_SDA, PIN_SCL, INA_ADDR);

  // Electrical triage before the bus owns the pins (pu=1,pd=1 => held high by module pull-ups).
  Wire.end();
  int sda_pu = read_pin(PIN_SDA, true), sda_pd = read_pin(PIN_SDA, false);
  int scl_pu = read_pin(PIN_SCL, true), scl_pd = read_pin(PIN_SCL, false);
  Serial.printf("PINTEST SDA pu=%d pd=%d  SCL pu=%d pd=%d\n", sda_pu, sda_pd, scl_pu, scl_pd);

  Wire.begin(PIN_SDA, PIN_SCL, 100000);
  int found = 0;
  Serial.print("SCAN:");
  for (uint8_t a = 0x08; a <= 0x77; ++a) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { Serial.printf(" 0x%02X", a); found++; }
  }
  Serial.printf("  (n=%d)\n", found);

  bool ok = false;
  uint16_t por = read_reg(0x00, &ok);
  Serial.printf("CONFIG_before_begin ok=%d value=0x%04X\n", ok, por);

  if (!ina.begin(&Wire)) {
    Serial.println("INA219 begin FAILED (no ACK at 0x40)");
    Serial.println("=== T22A_DIAG end ===");
    return;
  }
  ina.setCalibration_32V_2A();
  uint16_t cfg = read_reg(0x00, &ok);
  uint16_t cal = read_reg(0x05, &ok);
  Serial.printf("CONFIG_after_cal=0x%04X CAL=0x%04X\n", cfg, cal);
  delay(100);

  Serial.println("i,ms,bus_V,shunt_mV,current_mA,power_mW,raw_cur");
  for (int i = 0; i < 20; ++i) {
    float bus = ina.getBusVoltage_V();
    float sh  = ina.getShuntVoltage_mV();
    float cur = ina.getCurrent_mA();
    float pw  = ina.getPower_mW();
    int16_t rc = (int16_t)read_reg(0x04, &ok);
    Serial.printf("%d,%lu,%.3f,%.2f,%.1f,%.0f,%d\n", i, (unsigned long)millis(), bus, sh, cur, pw, rc);
    delay(100);
  }
  Serial.println("=== T22A_DIAG end ===");
}

void setup() {
  Serial.begin(115200);
  delay(1500);
}

void loop() {
  report();
  delay(10000);
}

// fw_replay.ino — T22a B1: replay G3 open schedules on the ESP32-S3 and log the board's own supply current
// through the INA219 shunt (SDA=GPIO8, SCL=GPIO9, 0x40). One mode only: replay -> RAM -> flash -> LED.
//
// Boot sequence
//   * /armed absent                -> command mode (STATUS, DUMP, CLEAR, ARM, DISARM, INA, LED). LED off.
//   * /armed present, no results   -> radios off, load every schedule into RAM, sample 10 Hz for warmup_s.
//       mean shunt current < min_shunt_ma -> the board is fed by USB, not through the shunt: do NOT measure,
//       stay armed, command mode, blue blip every 2 s ("waiting for the charger").
//       otherwise run all schedules back to back, idle_s of idle after each one (baseline), then write
//       trace_<tag>.csv + trace__warmup.csv + meta.json, delete /armed, LED green blinking 3 Hz = DONE.
//   * any failure -> /error.txt (if the FS is up) and LED solid red.
//
// Measurement-path rules (memory deep3-c2b-ina219-path / C2B_REPORT): store raw CURRENT (0.1 mA LSB, CAL 4096 on
// the 0.1 ohm shunt), raw SHUNT (10 uV LSB) and raw BUS registers; the POWER register (2 mW LSB) is not used.
// INA219 config 0x39FF: 32 V range, PGA /8, shunt ADC 128-sample average (68.1 ms), bus ADC 12-bit single.
//
// Timing: frame period = 1000 / fps_play ms on FreeRTOS ticks (1 kHz). Open frame = busy integer MAC loop for
// busy_ms (real computation, checked against esp_timer), closed frame = vTaskDelayUntil, i.e. the core idles in
// the FreeRTOS idle task (WAITI). Light sleep is NOT used: it would stop core 0 and with it the 10 Hz sampler.
// Samples are buffered in PSRAM (16 B each); flash is touched only after the last idle period.
// No Serial output inside the measured window.
#include <Arduino.h>
#include <Wire.h>
#include <LittleFS.h>
#include <ArduinoJson.h>
#include <WiFi.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"

#define FW_VERSION "fw_replay T22a-1"
static const int PIN_SDA = 8;
static const int PIN_SCL = 9;
static const uint8_t INA_ADDR = 0x40;
static const int LED_PIN = 48;            // on-board WS2812 of the ESP32-S3-CAM
static const uint16_t INA_CONFIG = 0x39FF;
static const uint16_t INA_CAL = 4096;
static const int MAX_SCHED = 8;

enum Phase : uint8_t { PH_WARMUP = 0, PH_PLAY = 1, PH_IDLE = 2 };
enum State { ST_CMD, ST_WAIT_CHARGER, ST_DONE, ST_ERROR };

struct Sample {             // 16 bytes
  uint32_t t_ms;
  uint32_t frame;           // frame index of the schedule being replayed (n during the idle tail)
  int16_t cur_raw;          // reg 0x04, 0.1 mA / LSB
  int16_t shunt_raw;        // reg 0x01, 10 uV / LSB
  uint16_t bus_raw;         // reg 0x02 as read (bits 15:3 = 4 mV LSB, bit1 CNVR, bit0 OVF)
  uint8_t phase;
  uint8_t sched;            // schedule index, 255 = warmup
};
static_assert(sizeof(Sample) == 16, "Sample must be 16 bytes");

struct Sched {
  String tag, file;
  uint32_t n = 0, n_open_cfg = 0, n_open_file = 0;
  float fps_native = 0;
  uint8_t *a = nullptr;
  uint32_t t_start_ms = 0, t_end_ms = 0, t_idle_end_ms = 0;
  uint32_t opens_exec = 0, overruns = 0;
  int64_t busy_us_sum = 0;
};

// ---- configuration (config.json) ----
static float cfg_fps = 12.5f;
static uint32_t cfg_busy_ms = 72, cfg_sample_hz = 10, cfg_warmup_s = 10, cfg_idle_s = 20;
static float cfg_min_shunt_ma = 15.0f, cfg_expected_s = 0;
static String cfg_version, cfg_gate;
static Sched sch[MAX_SCHED];
static int n_sched = 0;

// ---- shared with the sampler task ----
static Sample *g_buf = nullptr;
static uint32_t g_cap = 0;
static volatile uint32_t g_n = 0, g_overflow = 0, g_i2c_err = 0;
static volatile uint32_t g_frame = 0;
static volatile uint8_t g_phase = PH_WARMUP, g_sched = 255;
static volatile bool g_sampling = false;

static State g_state = ST_CMD;
static String g_err;
static float g_warm_ma = NAN;
static uint32_t g_t_boot_ms = 0;

// ---------------------------------------------------------------- INA219
static bool ina_write(uint8_t reg, uint16_t v) {
  Wire.beginTransmission(INA_ADDR);
  Wire.write(reg);
  Wire.write((uint8_t)(v >> 8));
  Wire.write((uint8_t)(v & 0xFF));
  return Wire.endTransmission() == 0;
}

static bool ina_read(uint8_t reg, uint16_t *v) {
  Wire.beginTransmission(INA_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)INA_ADDR, 2) != 2) return false;
  *v = ((uint16_t)Wire.read() << 8) | (uint16_t)Wire.read();
  return true;
}

static bool ina_init(String &why) {
  Wire.begin(PIN_SDA, PIN_SCL, 100000);
  Wire.setTimeOut(20);
  uint16_t v = 0;
  if (!ina_read(0x00, &v)) { why = "INA219 no ACK at 0x40"; return false; }
  if (!ina_write(0x00, INA_CONFIG) || !ina_read(0x00, &v) || v != INA_CONFIG) {
    why = "INA219 config verify failed"; return false;
  }
  // A silently failed CALIBRATION write makes CURRENT read a constant zero, which looks like data: verify it.
  if (!ina_write(0x05, INA_CAL) || !ina_read(0x05, &v) || v != INA_CAL) {
    why = "INA219 calibration verify failed"; return false;
  }
  return true;
}

static bool ina_sample(Sample &s) {
  uint16_t sh = 0, bus = 0, cur = 0;
  bool ok = ina_read(0x01, &sh) && ina_read(0x02, &bus) && ina_read(0x04, &cur);
  s.shunt_raw = (int16_t)sh;
  s.bus_raw = bus;
  s.cur_raw = (int16_t)cur;
  return ok;
}

static void sampler_task(void *) {
  const TickType_t per = pdMS_TO_TICKS(1000 / cfg_sample_hz);
  TickType_t last = xTaskGetTickCount();
  for (;;) {
    xTaskDelayUntil(&last, per);
    if (!g_sampling) continue;
    Sample s;
    s.t_ms = millis();
    s.frame = g_frame;
    s.phase = g_phase;
    s.sched = g_sched;
    if (!ina_sample(s)) { g_i2c_err = g_i2c_err + 1; continue; }
    uint32_t n = g_n;
    if (n < g_cap) { g_buf[n] = s; g_n = n + 1; }
    else g_overflow = g_overflow + 1;
  }
}

// ---------------------------------------------------------------- LED
static void led(uint8_t r, uint8_t g, uint8_t b) { rgbLedWrite(LED_PIN, r, g, b); }

static void led_service() {
  static uint32_t last = 0;
  static bool on = false;
  uint32_t now = millis();
  switch (g_state) {
    case ST_DONE:            // 3 blinks per second
      if (now - last >= 167) { last = now; on = !on; on ? led(0, 40, 0) : led(0, 0, 0); }
      break;
    case ST_ERROR:
      led(40, 0, 0);
      break;
    case ST_WAIT_CHARGER:    // short blue blip every 2 s
      if (!on && now - last >= 2000) { last = now; on = true; led(0, 0, 40); }
      if (on && now - last >= 60) { on = false; led(0, 0, 0); }
      break;
    default:
      break;
  }
}

// ---------------------------------------------------------------- busy work (stand-in for one inference)
static int8_t wA[4096], wB[4096];
static volatile int32_t g_sink = 0;

static void busy_until(int64_t t_end_us) {
  int32_t acc = 0;
  uint32_t j = 0;
  while (esp_timer_get_time() < t_end_us) {
    for (int i = 0; i < 256; ++i) acc += (int32_t)wA[(j + i) & 4095] * wB[(i * 17 + j) & 4095];
    j += 257;
  }
  g_sink += acc;
}

// ---------------------------------------------------------------- files
static void fail(const String &why) {
  g_sampling = false;
  g_state = ST_ERROR;
  g_err = why;
  File f = LittleFS.open("/error.txt", "w");
  if (f) { f.printf("%s\nmillis=%lu\n", why.c_str(), (unsigned long)millis()); f.close(); }
  Serial.printf("ERROR %s\n", why.c_str());
}

static bool results_exist() {
  if (LittleFS.exists("/meta.json")) return true;
  File root = LittleFS.open("/");
  for (File f = root.openNextFile(); f; f = root.openNextFile()) {
    if (String(f.name()).startsWith("trace_")) return true;
  }
  return false;
}

static bool load_config(String &why) {
  File f = LittleFS.open("/config.json", "r");
  if (!f) { why = "config.json missing"; return false; }
  JsonDocument d;
  DeserializationError e = deserializeJson(d, f);
  f.close();
  if (e) { why = String("config.json parse: ") + e.c_str(); return false; }
  cfg_version = d["version"] | "";
  cfg_gate = d["gate"] | "";
  cfg_fps = d["fps_play"] | 12.5f;
  cfg_busy_ms = d["busy_ms"] | 72;
  cfg_sample_hz = d["sample_hz"] | 10;
  cfg_warmup_s = d["warmup_s"] | 10;
  cfg_idle_s = d["idle_s"] | 20;
  cfg_min_shunt_ma = d["min_shunt_ma"] | 15.0f;
  cfg_expected_s = d["expected_total_s"] | 0.0f;
  n_sched = 0;
  for (JsonObject s : d["schedules"].as<JsonArray>()) {
    if (n_sched >= MAX_SCHED) { why = "too many schedules"; return false; }
    Sched &c = sch[n_sched++];
    c.tag = (const char *)(s["tag"] | "");
    c.file = (const char *)(s["file"] | "");
    c.n = s["n"] | 0;
    c.n_open_cfg = s["n_open"] | 0;
    c.fps_native = s["fps_native"] | 0.0f;
  }
  uint32_t period = (uint32_t)lroundf(1000.0f / cfg_fps);
  if (n_sched == 0) { why = "no schedules"; return false; }
  if (fabsf(period * cfg_fps - 1000.0f) > 1e-3f) { why = "1000/fps_play must be an integer ms"; return false; }
  if (cfg_busy_ms + 4 > period) { why = "busy_ms does not fit in one frame period"; return false; }
  if (1000 % cfg_sample_hz) { why = "sample_hz must divide 1000"; return false; }
  return true;
}

// Parse sched CSV (header frame,a_t) into c.a; frame indices must be 0..n-1 in order.
static bool load_schedule(Sched &c, String &why) {
  File f = LittleFS.open(c.file, "r");
  if (!f) { why = "missing " + c.file; return false; }
  free(c.a);
  c.a = (uint8_t *)heap_caps_malloc(c.n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!c.a) { why = "no RAM for " + c.tag; f.close(); return false; }
  f.readStringUntil('\n');                     // header
  uint32_t k = 0;
  c.n_open_file = 0;
  while (f.available()) {
    String line = f.readStringUntil('\n');
    line.trim();
    if (!line.length()) continue;
    int comma = line.indexOf(',');
    if (comma < 0 || (uint32_t)line.substring(0, comma).toInt() != k || k >= c.n) {
      why = c.tag + ": bad line " + String(k); f.close(); return false;
    }
    c.a[k] = line.substring(comma + 1).toInt() ? 1 : 0;
    c.n_open_file += c.a[k];
    k++;
  }
  f.close();
  if (k != c.n) { why = c.tag + ": " + String(k) + " frames, config says " + String(c.n); return false; }
  if (c.n_open_file != c.n_open_cfg) { why = c.tag + ": open count mismatch"; return false; }
  return true;
}

static bool write_traces(String &why) {
  char line[96];
  for (int si = -1; si < n_sched; ++si) {
    String name = si < 0 ? String("/trace__warmup.csv") : "/trace_" + sch[si].tag + ".csv";
    File f = LittleFS.open(name, "w");
    if (!f) { why = "cannot write " + name; return false; }
    f.print("t_ms,frame,phase,cur_raw,shunt_raw,bus_raw\n");
    uint8_t want = si < 0 ? 255 : (uint8_t)si;
    for (uint32_t i = 0; i < g_n; ++i) {
      const Sample &s = g_buf[i];
      if (s.sched != want) continue;
      int len = snprintf(line, sizeof(line), "%lu,%lu,%u,%d,%d,%u\n", (unsigned long)s.t_ms,
                         (unsigned long)s.frame, s.phase, s.cur_raw, s.shunt_raw, s.bus_raw);
      if (f.write((const uint8_t *)line, len) != (size_t)len) { f.close(); why = "write failed " + name; return false; }
    }
    f.close();
  }
  return true;
}

static bool write_meta(String &why) {
  JsonDocument d;
  d["fw"] = FW_VERSION;
  d["config_version"] = cfg_version;
  d["gate"] = cfg_gate;
  d["boot_millis_at_run_start"] = g_t_boot_ms;
  d["note_clock"] = "no RTC: times are millis() since boot; the host records wall-clock time at DUMP";
  d["fps_play"] = cfg_fps;
  d["busy_ms"] = cfg_busy_ms;
  d["sample_hz"] = cfg_sample_hz;
  d["warmup_s"] = cfg_warmup_s;
  d["idle_s"] = cfg_idle_s;
  d["ina219_config"] = INA_CONFIG;
  d["ina219_cal"] = INA_CAL;
  d["current_lsb_mA"] = 0.1;
  d["shunt_lsb_uV"] = 10;
  d["bus_lsb_mV"] = 4;
  d["pins"] = "SDA=8 SCL=9 addr=0x40";
  d["warmup_mean_mA"] = g_warm_ma;
  d["samples_total"] = g_n;
  d["samples_overflow"] = g_overflow;
  d["i2c_errors"] = g_i2c_err;
  d["sink"] = g_sink;
  JsonArray arr = d["schedules"].to<JsonArray>();
  for (int i = 0; i < n_sched; ++i) {
    JsonObject o = arr.add<JsonObject>();
    const Sched &c = sch[i];
    uint32_t ns = 0;
    for (uint32_t k = 0; k < g_n; ++k) ns += g_buf[k].sched == i;
    o["tag"] = c.tag;
    o["n_frames"] = c.n;
    o["fps_native"] = c.fps_native;
    o["n_open_schedule"] = c.n_open_file;
    o["n_open_executed"] = c.opens_exec;
    o["overruns"] = c.overruns;
    o["busy_ms_mean"] = c.opens_exec ? (double)c.busy_us_sum / c.opens_exec / 1000.0 : 0.0;
    o["t_start_ms"] = c.t_start_ms;
    o["t_end_ms"] = c.t_end_ms;
    o["t_idle_end_ms"] = c.t_idle_end_ms;
    o["samples"] = ns;
  }
  File f = LittleFS.open("/meta.json", "w");
  if (!f) { why = "cannot write meta.json"; return false; }
  serializeJson(d, f);
  f.close();
  return true;
}

// ---------------------------------------------------------------- the measurement run
static void run_measurement() {
  String why;
  WiFi.mode(WIFI_OFF);
  btStop();
  led(0, 0, 0);
  for (int i = 0; i < 4096; ++i) { wA[i] = (int8_t)(esp_random() & 0xFF); wB[i] = (int8_t)(esp_random() & 0xFF); }
  for (int i = 0; i < n_sched; ++i)
    if (!load_schedule(sch[i], why)) { fail(why); return; }

  float total = cfg_warmup_s;
  for (int i = 0; i < n_sched; ++i) total += sch[i].n / cfg_fps + cfg_idle_s;
  g_cap = (uint32_t)((total + 120.0f) * cfg_sample_hz);
  g_buf = (Sample *)heap_caps_malloc((size_t)g_cap * sizeof(Sample), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!g_buf) { fail("PSRAM alloc failed"); return; }
  g_n = 0;

  // Warmup: sample only, then decide whether the shunt carries the board current.
  g_t_boot_ms = millis();
  g_sched = 255; g_phase = PH_WARMUP; g_frame = 0;
  g_sampling = true;
  vTaskDelay(pdMS_TO_TICKS(cfg_warmup_s * 1000));
  double acc = 0; uint32_t m = 0;
  for (uint32_t i = 0; i < g_n; ++i) if (g_buf[i].sched == 255) { acc += g_buf[i].cur_raw * 0.1; m++; }
  g_warm_ma = m ? (float)(acc / m) : NAN;
  if (!(m > 0 && g_warm_ma >= cfg_min_shunt_ma)) {
    g_sampling = false;
    free(g_buf); g_buf = nullptr;
    g_state = ST_WAIT_CHARGER;
    Serial.printf("ARMED but warmup mean %.1f mA < %.1f mA over %lu samples: board not fed through the shunt "
                  "(USB?). Not measuring; still armed.\n", g_warm_ma, cfg_min_shunt_ma, (unsigned long)m);
    return;
  }

  const TickType_t period = pdMS_TO_TICKS((uint32_t)lroundf(1000.0f / cfg_fps));
  const int64_t busy_us = (int64_t)cfg_busy_ms * 1000;
  for (int i = 0; i < n_sched; ++i) {
    Sched &c = sch[i];
    g_frame = 0; g_sched = (uint8_t)i; g_phase = PH_PLAY;
    c.t_start_ms = millis();
    TickType_t last = xTaskGetTickCount();
    for (uint32_t k = 0; k < c.n; ++k) {
      g_frame = k;
      if (c.a[k]) {
        int64_t t0 = esp_timer_get_time();
        busy_until(t0 + busy_us);
        c.busy_us_sum += esp_timer_get_time() - t0;
        c.opens_exec++;
      }
      if (xTaskDelayUntil(&last, period) == pdFALSE) c.overruns++;
    }
    c.t_end_ms = millis();
    g_phase = PH_IDLE; g_frame = c.n;
    vTaskDelay(pdMS_TO_TICKS(cfg_idle_s * 1000));
    c.t_idle_end_ms = millis();
  }
  g_sampling = false;
  vTaskDelay(pdMS_TO_TICKS(200));             // let an in-flight sample finish

  if (!write_traces(why) || !write_meta(why)) { fail(why); return; }
  LittleFS.remove("/armed");
  free(g_buf); g_buf = nullptr;
  g_state = ST_DONE;
}

// ---------------------------------------------------------------- serial commands
static uint32_t crc32_file(File &f) {
  uint32_t crc = 0xFFFFFFFF;
  uint8_t b[512];
  while (f.available()) {
    int n = f.read(b, sizeof(b));
    for (int i = 0; i < n; ++i) {
      crc ^= b[i];
      for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
  }
  return ~crc;
}

static void cmd_status() {
  Serial.printf("STATUS fw=%s state=%s armed=%d results=%d err=%s\n", FW_VERSION,
                g_state == ST_CMD ? "CMD" : g_state == ST_WAIT_CHARGER ? "WAIT_CHARGER" :
                g_state == ST_DONE ? "DONE" : "ERROR",
                LittleFS.exists("/armed"), results_exist(), g_err.c_str());
  Serial.printf("config version=%s gate=%s fps_play=%.2f busy_ms=%lu sample_hz=%lu warmup_s=%lu idle_s=%lu "
                "min_shunt_ma=%.1f expected_total_s=%.0f\n", cfg_version.c_str(), cfg_gate.c_str(), cfg_fps,
                (unsigned long)cfg_busy_ms, (unsigned long)cfg_sample_hz, (unsigned long)cfg_warmup_s,
                (unsigned long)cfg_idle_s, cfg_min_shunt_ma, cfg_expected_s);
  for (int i = 0; i < n_sched; ++i) {
    String why;
    bool ok = load_schedule(sch[i], why);
    Serial.printf("sched %d tag=%s file=%s n=%lu n_open_cfg=%lu n_open_file=%lu fps_native=%.1f load=%s\n", i,
                  sch[i].tag.c_str(), sch[i].file.c_str(), (unsigned long)sch[i].n,
                  (unsigned long)sch[i].n_open_cfg, (unsigned long)sch[i].n_open_file, sch[i].fps_native,
                  ok ? "OK" : why.c_str());
    free(sch[i].a); sch[i].a = nullptr;
  }
  Serial.printf("fs used=%u total=%u psram_free=%u\n", (unsigned)LittleFS.usedBytes(), (unsigned)LittleFS.totalBytes(),
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  File root = LittleFS.open("/");
  for (File f = root.openNextFile(); f; f = root.openNextFile())
    Serial.printf("file %s %u\n", f.name(), (unsigned)f.size());
  Serial.println("STATUS_END");
}

static void cmd_dump() {
  File root = LittleFS.open("/");
  int n = 0;
  for (File f = root.openNextFile(); f; f = root.openNextFile()) n++;
  Serial.printf("DUMP_BEGIN %d\n", n);
  root = LittleFS.open("/");
  for (File f = root.openNextFile(); f; f = root.openNextFile()) {
    String name = String("/") + f.name();
    f.close();
    File g = LittleFS.open(name, "r");
    uint32_t crc = crc32_file(g);
    size_t size = g.size();
    g.seek(0);
    Serial.printf("FILE %s %u %08lx\n", name.c_str() + 1, (unsigned)size, (unsigned long)crc);
    uint8_t b[512];
    while (g.available()) { int k = g.read(b, sizeof(b)); Serial.write(b, k); }
    g.close();
    Serial.print("\nENDFILE\n");
  }
  Serial.println("DUMP_END");
}

static void cmd_clear() {
  File root = LittleFS.open("/");
  String del[32];
  int nd = 0;
  for (File f = root.openNextFile(); f && nd < 32; f = root.openNextFile()) {
    String nm = f.name();
    if (nm.startsWith("trace_") || nm == "meta.json" || nm == "error.txt" || nm == "armed") del[nd++] = "/" + nm;
  }
  for (int i = 0; i < nd; ++i) LittleFS.remove(del[i]);
  if (g_state != ST_ERROR) g_state = ST_CMD;
  led(0, 0, 0);
  Serial.printf("CLEARED %d file(s)\n", nd);
}

static void cmd_ina() {
  for (int i = 0; i < 5; ++i) {
    Sample s;
    bool ok = ina_sample(s);
    Serial.printf("INA ok=%d bus_V=%.3f shunt_mV=%.2f current_mA=%.1f\n", ok, (s.bus_raw >> 3) * 0.004,
                  s.shunt_raw * 0.01, s.cur_raw * 0.1);
    delay(150);
  }
}

static void handle(String c) {
  c.trim();
  c.toUpperCase();
  if (c == "STATUS") cmd_status();
  else if (c == "DUMP") cmd_dump();
  else if (c == "CLEAR") cmd_clear();
  else if (c == "INA") cmd_ina();
  else if (c == "ARM") {
    if (results_exist()) Serial.println("ARM refused: results present, DUMP then CLEAR first");
    else { File f = LittleFS.open("/armed", "w"); f.print("1"); f.close(); Serial.println("ARMED (runs at next boot)"); }
  } else if (c == "DISARM") { LittleFS.remove("/armed"); if (g_state == ST_WAIT_CHARGER) g_state = ST_CMD; led(0, 0, 0); Serial.println("DISARMED"); }
  else if (c == "LED") { for (int i = 0; i < 6; ++i) { led(0, 40, 0); delay(167); led(0, 0, 0); delay(167); } Serial.println("LED_TEST_DONE"); }
  else if (c.length()) Serial.printf("? unknown '%s' (STATUS DUMP CLEAR ARM DISARM INA LED)\n", c.c_str());
}

// ---------------------------------------------------------------- setup / loop
void setup() {
  Serial.begin(115200);
  led(0, 0, 0);
  String why;
  if (!LittleFS.begin(false)) { g_state = ST_ERROR; g_err = "LittleFS mount failed"; Serial.println(g_err); return; }
  if (!load_config(why)) { fail(why); return; }
  if (!ina_init(why)) { fail(why); return; }
  xTaskCreatePinnedToCore(sampler_task, "ina_sampler", 4096, nullptr, 5, nullptr, 0);
  if (LittleFS.exists("/armed") && !results_exist()) run_measurement();
  else if (results_exist()) g_state = ST_DONE;
  Serial.printf("READY %s state=%d (send STATUS)\n", FW_VERSION, (int)g_state);
}

void loop() {
  led_service();
  if (Serial.available()) handle(Serial.readStringUntil('\n'));
  delay(10);
}

# Camera settings patch: change exposure, gain, brightness and contrast from the laptop

After this one flash, `laptop/camera_settings.py` can read and change the camera chip settings
while the drone is running, and `run_baseline.py` applies the ones in `config.yaml`
(`camera: sensor:`) at the start of every run. No further reflashing is needed to tune them.

The firmware today has no way to do this: it sets `brightness +1` once at boot (in
`at_detect.c`) and never touches the sensor again.

## Which source to patch

Patch **the firmware source drone 5 was last flashed from**: the one with the phone-hotspot Wi-Fi
settings and the 125-byte telemetry (z, velocities, quaternion). The copy in
`esp-everything-qgc-bridge-main\main` on your laptop is an older version set up for drone 22 on
the TP-Link router. Flashing that copy would cut drone 5 off the hotspot and drop the attitude
telemetry the optimisation needs.

## Steps

1. Copy `main/camera_settings.c` and `main/camera_settings.h` from this folder into the firmware's
   `main/` folder.
2. In `main/wifi_task.c` make two edits (`camera_settings.diff` shows them in context):
   - under `#include "wifi_task.h"` add
     ```c
     #include "camera_settings.h"
     ```
   - in the command loop, just **before** `} else if (len >= (int)sizeof(wifi_cmd_pkt_t)) {` add
     ```c
                } else if (cmd_buf[1] == CMD_CAMERA_SETTINGS) {
                    camera_settings_handle(cmd_buf, len, rx_sock, &source);
     ```
3. In `main/CMakeLists.txt` add `"camera_settings.c"` to `SRCS` (for example after `"camera_stream.c"`).
4. Build and flash drone 5 exactly as it was flashed last time (same sdkconfig: drone ID 5,
   hotspot Wi-Fi), e.g. `idf.py build` then `idf.py -p <port> flash`.
5. Check from the laptop folder, with the drone on the hotspot:
   ```
   python camera_settings.py --esp-ip 172.20.10.5
   python show_telemetry.py ...        (still 125-byte telemetry)
   ```
   The first command prints the sensor type, live exposure and gain, and every setting.

## Notes

- Command byte `0x0A` (`CMD_CAMERA_SETTINGS`). If that firmware version already uses `0x0A` for
  something else, change it in both `main/camera_settings.h` and `laptop/camera_settings.py`.
- The reply goes back to whatever address and port sent the request, so it works alongside
  `run_baseline.py`, `tag_stream.py` and QGroundControl.
- Settings are not stored: a reboot goes back to the firmware defaults. `run_baseline.py`
  re-applies the `config.yaml` ones at the start of every run.
- `vflip` / `hmirror` are reported but cannot be changed: the tag geometry depends on them.
- Firmware **without** this patch ignores the request (the laptop sends at most 5 changes per
  datagram, so the packet stays too short for the old generic command handler). `run_baseline.py`
  then logs "camera settings NOT applied" and flies as normal.
- Both camera users see the change: the drone's own AprilTag detector and the laptop stream.

## Host test (optional, no ESP-IDF needed)

`host_test/host_server.c` runs the real `camera_settings.c` against a fake OV3660 on
`127.0.0.1:15006`, so `camera_settings.py --esp-ip 127.0.0.1 --port 15006` can be tried on a PC
(Linux/macOS with gcc; needs the esp32-camera `driver/include` headers and small stubs for
`esp_err.h`, `esp_log.h`, `driver/ledc.h`, `sdkconfig.h`, `img_converters.h`).

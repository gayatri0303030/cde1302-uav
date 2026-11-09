#pragma once

/* Run-time camera sensor settings (exposure, gain, brightness, contrast, ...).
 *
 * The laptop (laptop/camera_settings.py, run_baseline.py --sensor) sends
 *
 *   request: pkt (0x02), cmd (CMD_CAMERA_SETTINGS), op (0 = read, 1 = change),
 *            n, n x { id (u8), value (i16, little-endian) }
 *
 * to the command port.  Changes are applied in the order given, then the
 * current settings go back to the sender's own address and port:
 *
 *   reply:   "ECSS", version (1), drone_id, sensor_pid (u16),
 *            live_exposure (i32), live_gain_x16 (i32), exposure_max (i32),
 *            n_params, n_results,
 *            n_params  x { id (u8), value (i16) },
 *            n_results x { id (u8), status (i8): 0 ok, -1 sensor refused, -2 unknown id }
 *
 * live_* are read from the sensor registers (OV3660/OV5640), so they show what
 * auto exposure is doing right now; -1 when not available.  Settings are not
 * saved: a reboot goes back to the firmware defaults (run_baseline.py applies
 * the configured ones at the start of every run).
 *
 * Both camera consumers see the change: the drone's own AprilTag detector and
 * the laptop stream.  Image orientation (vflip / hmirror) is reported but can
 * not be changed here, because the tag geometry depends on it.
 */

#include <stdint.h>
#include <netinet/in.h>

#define CMD_CAMERA_SETTINGS 0x0A   /* must match laptop/camera_settings.py */

/* Parameter ids (must match laptop/camera_settings.py). */
enum {
    CAM_P_BRIGHTNESS  = 1,   /* -3..3 on the OV3660 (firmware boots with +1)  */
    CAM_P_CONTRAST    = 2,   /* -3..3                                          */
    CAM_P_SHARPNESS   = 3,
    CAM_P_DENOISE     = 4,
    CAM_P_AE_LEVEL    = 5,   /* auto-exposure target, -5..5 (lower = darker)   */
    CAM_P_AEC         = 6,   /* 1 = auto exposure, 0 = manual (use aec_value)  */
    CAM_P_AEC_VALUE   = 7,   /* manual exposure time (sensor lines)            */
    CAM_P_AGC         = 8,   /* 1 = auto gain, 0 = manual (use agc_gain)       */
    CAM_P_AGC_GAIN    = 9,   /* manual gain, 0..64                             */
    CAM_P_GAINCEILING = 10,  /* auto-gain ceiling (raw sensor value)           */
    CAM_P_AEC2        = 11,  /* night-mode / DSP AEC on-off                    */
    CAM_P_LENC        = 12,  /* lens shading correction on-off                 */
    CAM_P_BPC         = 13,  /* black pixel correction on-off                  */
    CAM_P_WPC         = 14,  /* white pixel correction on-off                  */
    CAM_P_RAW_GMA     = 15,  /* gamma on-off                                   */
    CAM_P_VFLIP       = 16,  /* read only                                      */
    CAM_P_HMIRROR     = 17,  /* read only                                      */
};

/* Handles one CMD_CAMERA_SETTINGS datagram received on `sock` from `source`
 * and sends the reply back on the same socket.  Called by wifi_task. */
void camera_settings_handle(const uint8_t *buf, int len, int sock,
                            const struct sockaddr_in *source);

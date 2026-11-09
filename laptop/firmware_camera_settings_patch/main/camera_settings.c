/* Run-time camera sensor settings over the command port.  See camera_settings.h. */
#include "camera_settings.h"

#include "esp_camera.h"
#include "esp_log.h"
#include "sdkconfig.h"

#include <string.h>
#include <sys/socket.h>

static const char *TAG = "cam_settings";

#define OP_READ    0
#define OP_CHANGE  1
#define MAX_ITEMS  24

typedef struct __attribute__((packed)) {
    uint8_t  magic[4];        /* "ECSS" */
    uint8_t  version;         /* 1 */
    uint8_t  drone_id;
    uint16_t sensor_pid;
    int32_t  live_exposure;   /* exposure the sensor is using now (lines), -1 = n/a */
    int32_t  live_gain_x16;   /* gain the sensor is using now, x16, -1 = n/a       */
    int32_t  exposure_max;    /* longest exposure for this frame timing, -1 = n/a   */
    uint8_t  n_params;
    uint8_t  n_results;
} reply_hdr_t;

typedef struct __attribute__((packed)) { uint8_t id; int16_t value; } item_t;
typedef struct __attribute__((packed)) { uint8_t id; int8_t status; } result_t;

_Static_assert(sizeof(reply_hdr_t) == 22, "must match laptop/camera_settings.py");
_Static_assert(sizeof(item_t) == 3, "must match laptop/camera_settings.py");

static const uint8_t k_report_ids[] = {
    CAM_P_BRIGHTNESS, CAM_P_CONTRAST, CAM_P_SHARPNESS, CAM_P_DENOISE, CAM_P_AE_LEVEL,
    CAM_P_AEC, CAM_P_AEC_VALUE, CAM_P_AGC, CAM_P_AGC_GAIN, CAM_P_GAINCEILING,
    CAM_P_AEC2, CAM_P_LENC, CAM_P_BPC, CAM_P_WPC, CAM_P_RAW_GMA, CAM_P_VFLIP, CAM_P_HMIRROR,
};

static bool has_omni_regs(const sensor_t *s)
{
    return s->id.PID == OV3660_PID || s->id.PID == OV5640_PID;
}

static int reg8(sensor_t *s, int reg)
{
    return s->get_reg ? s->get_reg(s, reg, 0xFF) : -1;
}

/* Apply one change.  0 = ok, -1 = the sensor refused / has no such control,
 * -2 = unknown or read-only id. */
static int apply(sensor_t *s, uint8_t id, int v)
{
#define CALL(fn, arg) ((s->fn) ? ((s->fn(s, (arg)) == 0) ? 0 : -1) : -1)
    switch (id) {
    case CAM_P_BRIGHTNESS:  return CALL(set_brightness, v);
    case CAM_P_CONTRAST:    return CALL(set_contrast, v);
    case CAM_P_SHARPNESS:   return CALL(set_sharpness, v);
    case CAM_P_DENOISE:     return CALL(set_denoise, v);
    case CAM_P_AE_LEVEL:    return CALL(set_ae_level, v);
    case CAM_P_AEC:         return CALL(set_exposure_ctrl, v ? 1 : 0);
    case CAM_P_AEC_VALUE:   return CALL(set_aec_value, v);
    case CAM_P_AGC:         return CALL(set_gain_ctrl, v ? 1 : 0);
    case CAM_P_AGC_GAIN:    return CALL(set_agc_gain, v);
    case CAM_P_GAINCEILING: return CALL(set_gainceiling, (gainceiling_t)v);
    case CAM_P_AEC2:        return CALL(set_aec2, v ? 1 : 0);
    case CAM_P_LENC:        return CALL(set_lenc, v ? 1 : 0);
    case CAM_P_BPC:         return CALL(set_bpc, v ? 1 : 0);
    case CAM_P_WPC:         return CALL(set_wpc, v ? 1 : 0);
    case CAM_P_RAW_GMA:     return CALL(set_raw_gma, v ? 1 : 0);
    default:                return -2;
    }
#undef CALL
}

static int16_t value_of(sensor_t *s, uint8_t id)
{
    const camera_status_t *st = &s->status;
    switch (id) {
    case CAM_P_BRIGHTNESS:  return st->brightness;
    case CAM_P_CONTRAST:    return st->contrast;
    case CAM_P_SHARPNESS:   return st->sharpness;
    case CAM_P_DENOISE:     return st->denoise;
    case CAM_P_AE_LEVEL:    return st->ae_level;
    case CAM_P_AEC:         return st->aec;
    case CAM_P_AEC_VALUE:   return (int16_t)st->aec_value;
    case CAM_P_AGC:         return st->agc;
    case CAM_P_AGC_GAIN:    return st->agc_gain;
    case CAM_P_GAINCEILING:
        if (has_omni_regs(s)) {             /* 10-bit raw value; status keeps 8 bits */
            int hi = reg8(s, 0x3A18), lo = reg8(s, 0x3A19);
            if (hi >= 0 && lo >= 0) return (int16_t)(((hi & 0x03) << 8) | lo);
        }
        return st->gainceiling;
    case CAM_P_AEC2:        return st->aec2;
    case CAM_P_LENC:        return st->lenc;
    case CAM_P_BPC:         return st->bpc;
    case CAM_P_WPC:         return st->wpc;
    case CAM_P_RAW_GMA:     return st->raw_gma;
    case CAM_P_VFLIP:       return st->vflip;
    case CAM_P_HMIRROR:     return st->hmirror;
    default:                return 0;
    }
}

static void read_live(sensor_t *s, reply_hdr_t *h)
{
    h->live_exposure = h->live_gain_x16 = h->exposure_max = -1;
    if (!has_omni_regs(s)) return;
    int a = reg8(s, 0x3500), b = reg8(s, 0x3501), c = reg8(s, 0x3502);
    if (a >= 0 && b >= 0 && c >= 0) {
        h->live_exposure = ((a & 0x0F) << 12) | ((b & 0xFF) << 4) | ((c & 0xF0) >> 4);
    }
    int ga = reg8(s, 0x350A), gb = reg8(s, 0x350B);
    if (ga >= 0 && gb >= 0) {
        h->live_gain_x16 = ((ga & 0x03) << 8) | (gb & 0xFF);
    }
    int va = reg8(s, 0x380E), vb = reg8(s, 0x380F);
    if (va >= 0 && vb >= 0) {
        h->exposure_max = ((va & 0xFF) << 8) | (vb & 0xFF);
    }
}

void camera_settings_handle(const uint8_t *buf, int len, int sock,
                            const struct sockaddr_in *source)
{
    if (len < 4) {
        ESP_LOGW(TAG, "CMD_CAMERA_SETTINGS truncated (%d B)", len);
        return;
    }
    uint8_t op = buf[2];
    int n = buf[3];
    if (n > MAX_ITEMS || len < 4 + n * (int)sizeof(item_t)) {
        ESP_LOGW(TAG, "CMD_CAMERA_SETTINGS bad length (%d B for %d items)", len, n);
        return;
    }
    sensor_t *s = esp_camera_sensor_get();

    uint8_t out[sizeof(reply_hdr_t) + sizeof(k_report_ids) * sizeof(item_t)
                + MAX_ITEMS * sizeof(result_t)];
    reply_hdr_t h = {
        .magic = {'E', 'C', 'S', 'S'},
        .version = 1,
        .drone_id = (uint8_t)CONFIG_DRONE_ID,
        .sensor_pid = s ? s->id.PID : 0,
    };
    result_t results[MAX_ITEMS];
    int n_results = 0;

    if (s != NULL && op == OP_CHANGE) {
        for (int i = 0; i < n; i++) {
            item_t it;
            memcpy(&it, buf + 4 + i * sizeof(item_t), sizeof(it));
            int rc = apply(s, it.id, it.value);
            results[n_results++] = (result_t){ .id = it.id, .status = (int8_t)rc };
            ESP_LOGI(TAG, "set id %u = %d -> %s", it.id, it.value,
                     rc == 0 ? "ok" : rc == -2 ? "unknown/read-only" : "refused");
        }
    }

    size_t off = sizeof(h);
    if (s != NULL) {
        read_live(s, &h);
        for (size_t i = 0; i < sizeof(k_report_ids); i++) {
            item_t it = { .id = k_report_ids[i], .value = value_of(s, k_report_ids[i]) };
            memcpy(out + off, &it, sizeof(it));
            off += sizeof(it);
        }
        h.n_params = (uint8_t)sizeof(k_report_ids);
    } else {
        h.live_exposure = h.live_gain_x16 = h.exposure_max = -1;
    }
    h.n_results = (uint8_t)n_results;
    memcpy(out + off, results, n_results * sizeof(result_t));
    off += n_results * sizeof(result_t);
    memcpy(out, &h, sizeof(h));

    if (sendto(sock, out, off, 0, (const struct sockaddr *)source, sizeof(*source)) < 0) {
        ESP_LOGW(TAG, "camera settings reply not sent");
    }
}

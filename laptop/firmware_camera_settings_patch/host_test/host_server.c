/* Host test: the real camera_settings.c behind a UDP socket, with a fake OV3660. */
#include "camera_settings.h"
#include "esp_camera.h"
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static sensor_t S;
static int regs[0x10000];
sensor_t *esp_camera_sensor_get(void) { return &S; }
static int rng(int v, int lo, int hi) { return v < lo || v > hi ? -1 : 0; }
static int set_brightness(sensor_t *s, int v) { if (rng(v,-3,3)) return -1; s->status.brightness = v; return 0; }
static int set_contrast(sensor_t *s, int v) { if (rng(v,-3,3)) return -1; s->status.contrast = v; return 0; }
static int set_sharpness(sensor_t *s, int v) { if (rng(v,-3,3)) return -1; s->status.sharpness = v; return 0; }
static int set_denoise(sensor_t *s, int v) { if (rng(v,0,8)) return -1; s->status.denoise = v; return 0; }
static int set_ae_level(sensor_t *s, int v) { if (rng(v,-5,5)) return -1; s->status.ae_level = v; return 0; }
static int set_exposure_ctrl(sensor_t *s, int v) { s->status.aec = v; return 0; }
static int set_gain_ctrl(sensor_t *s, int v) { s->status.agc = v; return 0; }
static int set_aec_value(sensor_t *s, int v) { if (v > 984) v = 984; s->status.aec_value = v;
    regs[0x3500] = (v >> 12) & 0x0F; regs[0x3501] = (v >> 4) & 0xFF; regs[0x3502] = (v << 4) & 0xF0; return 0; }
static int set_agc_gain(sensor_t *s, int v) { if (v > 64) v = 64; s->status.agc_gain = v;
    int g = v << 4; if (g) g -= 1; regs[0x350A] = g >> 8; regs[0x350B] = g & 0xFF; return 0; }
static int set_gainceiling(sensor_t *s, gainceiling_t v) { regs[0x3A18] = (v >> 8) & 3; regs[0x3A19] = v & 0xFF; s->status.gainceiling = v; return 0; }
static int set_aec2(sensor_t *s, int v) { s->status.aec2 = v; return 0; }
static int get_reg(sensor_t *s, int reg, int mask) { (void)s; return regs[reg] & mask; }

int main(int argc, char **argv) {
    int port = argc > 1 ? atoi(argv[1]) : 15006;
    S.id.PID = OV3660_PID; S.status.brightness = 1; S.status.saturation = -2; S.status.aec = 1; S.status.agc = 1;
    S.status.vflip = 1; S.status.lenc = 1; S.status.bpc = 1;
    S.set_brightness = set_brightness; S.set_contrast = set_contrast; S.set_sharpness = set_sharpness;
    S.set_denoise = set_denoise; S.set_ae_level = set_ae_level; S.set_exposure_ctrl = set_exposure_ctrl;
    S.set_gain_ctrl = set_gain_ctrl; S.set_aec_value = set_aec_value; S.set_agc_gain = set_agc_gain;
    S.set_gainceiling = set_gainceiling; S.set_aec2 = set_aec2; S.get_reg = get_reg;   /* lenc/bpc/wpc/raw_gma left NULL */
    regs[0x3500] = 0; regs[0x3501] = 0x2A; regs[0x3502] = 0x80;      /* live exposure 680 */
    regs[0x350A] = 0; regs[0x350B] = 0x40;                            /* gain 64/16 = x4 */
    regs[0x380E] = 0x03; regs[0x380F] = 0xD8;                         /* max 984 */
    regs[0x3A18] = 0x00; regs[0x3A19] = 0xF8;
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = htonl(INADDR_LOOPBACK) };
    if (bind(sock, (struct sockaddr *)&a, sizeof a) < 0) { perror("bind"); return 1; }
    printf("fake drone camera on 127.0.0.1:%d\n", port); fflush(stdout);
    for (;;) {
        uint8_t buf[1024]; struct sockaddr_in src; socklen_t sl = sizeof src;
        int len = recvfrom(sock, buf, sizeof buf, 0, (struct sockaddr *)&src, &sl);
        if (len < 2 || buf[0] != 0x02) continue;
        if (buf[1] == CMD_CAMERA_SETTINGS) camera_settings_handle(buf, len, sock, &src);
        fflush(stdout);
    }
}

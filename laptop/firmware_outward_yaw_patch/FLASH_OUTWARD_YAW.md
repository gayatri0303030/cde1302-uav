# Outward-yaw firmware patch (nose 90 deg right of travel for part of a trajectory)

Needed only for `outward_rectangle` paths (camera facing out of the loop).  Everything else flies
exactly as before, and old laptop commands send nothing new.

What it adds to CMD_TRAJ_START (on top of the yaw-follow patch you already flashed):
  flags bit1 + two uint16 point numbers FROM, TO.  For points FROM..TO-1 the nose points 90 deg to
  the RIGHT of the direction of travel (outward for the loop); outside them it follows the path
  (bit0) or holds its heading.  Same 1 rad/s turn limit.  Older firmware ignores bit1 and the extra
  bytes, so the loop is then flown nose-along-the-path (run_baseline then prints
  "outward nose: ... flash the outward-yaw firmware patch").

## Edit the source drone 5 was last flashed from (the one with the yaw-follow patch)

Three small edits.  `outward_yaw.diff` shows them in context; `main/` has the three whole files as
they are with ONLY the yaw-follow patch + this change.  If you also added the camera-settings
patch to `wifi_task.c`, do NOT copy `main/wifi_task.c` over yours: make edit 3 by hand.

1. `main/nav_task.h`: change the `nav_traj_start` line to
   ```c
   void nav_traj_start(uint8_t id, uint16_t dt_ms, uint8_t flags, uint16_t out_from, uint16_t out_to);
   ```
2. `main/nav_task.c` (copy the file, or make the 4 edits in the diff):
   - under `static bool s_traj_yaw_follow;` add
     ```c
     static uint16_t   s_traj_out_from;     /* points [from, to): nose 90 deg right of travel (flag bit1) */
     static uint16_t   s_traj_out_to;       /* from == to = no outward window                            */
     ```
   - in the trajectory tick replace `if (s_traj_yaw_follow) {` with
     ```c
     bool in_out = (i >= (int)s_traj_out_from && i < (int)s_traj_out_to);
     if (s_traj_yaw_follow || in_out) {
     ```
     and `float want = atan2f(dy, dx);` with
     ```c
     float want = atan2f(dy, dx) + (in_out ? 0.5f * (float)M_PI : 0.0f);
     ```
   - change the `void nav_traj_start(uint8_t id, uint16_t dt_ms, uint8_t flags)` line to the new
     signature (as in 1.) and under `s_traj_yaw_follow = (flags & 0x01) != 0;` add
     ```c
     s_traj_out_from   = (flags & 0x02) ? out_from : 0;
     s_traj_out_to     = (flags & 0x02) ? out_to   : 0;
     ```
3. `main/wifi_task.c`, in `handle_traj_start`, replace `nav_traj_start(buf[2], dt_ms, flags);` with
   ```c
   uint16_t out_from = 0, out_to = 0;            /* optional: bit1 = nose outward for points [from, to) */
   if (len >= 10) {
       memcpy(&out_from, buf + 6, sizeof(out_from));
       memcpy(&out_to,   buf + 8, sizeof(out_to));
   }
   nav_traj_start(buf[2], dt_ms, flags, out_from, out_to);
   ```

Build and flash exactly as before (same sdkconfig: drone ID 5, hotspot Wi-Fi):
```
idf.py build
idf.py -p COMx flash
```
Check: fly `--trajectory outward_rectangle 3.2 1.6`; after the path run_baseline prints
`outward nose: within NN deg of the planned outward heading - working`.

Not compiled here (no ESP-IDF in this sandbox): the edited files give the same compiler messages
as the unedited ones with stub headers, i.e. no new errors.  The laptop side (protocol.py,
run_baseline.py, make_trajectory.py) and the simulator (miniproject/fake_drone.py) already speak it.

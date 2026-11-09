"""Set the drone's camera mount ONCE after changing it physically; every later run uses it.

  python set_camera_mount.py --pitch 50                  lengthened mount (lens 9.5 cm ahead), 50 deg tilt
  python set_camera_mount.py --pitch 48 --fwd 0.095      any measured values
  python set_camera_mount.py --show                      print the mount runs use now, change nothing
  python set_camera_mount.py --reset                     back to the 29 Oct mount (7.5 cm, 65 deg)

--pitch  lens angle BELOW HORIZONTAL in degrees (90 = straight down; 29 Oct mount = 65,
         i.e. 25 deg from vertical).  Measure it on the drone; a wrong angle moves every tag.
--fwd    lens ahead of the PX4 (m).  Default 0.095 = the 29 Oct 7.5 cm + the 2 cm longer mount.
--down   lens below the PX4 (m).  Default 0.025 (unchanged).

Writes optimization_comparison/laptop/mapper_config_current.yaml (the 29 Oct file with the new
camera.extrinsic_T_B_C) and the same values into miniproject/config.yaml camera.mount.
run_baseline.py takes the mount from mapper_config_current.yaml when it exists, so the live tag
positions and the optimisation both use it; each run still saves its own mapper_config_used.yaml,
so earlier flights keep the mount they were flown with.  Check the 'camera mount' line at the
start of every run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE / "optimization_comparison" / "laptop"
BASE = OPT / "mapper_config_29_Oct.yaml"
CURRENT = OPT / "mapper_config_current.yaml"
CONFIG = HERE / "miniproject" / "config.yaml"


def extrinsic_block(pitch: float, fwd: float, down: float, mount_id: str, when: str) -> str:
    p = math.radians(pitch)
    s, c = math.sin(p), math.cos(p)
    return (
        "  extrinsic_T_B_C:\n"
        f"    # Set by set_camera_mount.py on {when}: camera optical centre {fwd * 100:.1f} cm AHEAD of and\n"
        f"    # {down * 100:.1f} cm BELOW the PX4 body origin, lens {pitch:.1f} deg below the nose direction\n"
        f"    # ({90 - pitch:.1f} deg from vertical).  Earlier mounts: mapper_config_29_Oct.yaml (7.5 cm, 65 deg).\n"
        "    # Body = FRD (x forward, y right, z down).  Camera = OpenCV optical (x image-right, y image-down,\n"
        "    # z out of the lens).  Columns of the matrix are the camera axes written in the body frame:\n"
        "    #   camera x (image right) = body +y (drone's right)\n"
        f"    #   camera y (image down)  = (-sin{pitch:g}, 0, cos{pitch:g})\n"
        f"    #   camera z (lens axis)   = ( cos{pitch:g}, 0, sin{pitch:g})\n"
        f"    mount_id: {mount_id}\n"
        f"    translation_m: [{fwd!r}, 0.0, {down!r}]\n"
        "    rotation_matrix:\n"
        f"    - [0.0, {-s!r}, {c!r}]\n"
        "    - [1.0, 0.0, 0.0]\n"
        f"    - [0.0, {c!r}, {s!r}]\n")


def replace_extrinsic(text: str, block: str) -> str:
    m = re.search(r"^  extrinsic_T_B_C:\n(?:    .*\n|\s*\n)*?    rotation_matrix:\n(?:    - \[.*\]\n){3}", text, re.M)
    if not m:
        raise SystemExit(f"{BASE.name}: camera.extrinsic_T_B_C block not found - not changing anything")
    out = text[:m.start()] + block
    rest = text[m.end():]
    return out + rest


def fix_descriptions(text: str, pitch: float, fwd: float, down: float, when: str) -> str:
    """The 29 Oct file's description lines after the matrix would still say 65 deg: rewrite them."""
    text = re.sub(r"^(    rotation_description:).*$",
                  rf'\g<1> "OpenCV optical (right/down/forward) to FRD body, {pitch:g} deg nose-down pitch '
                  rf'({90 - pitch:g} deg from vertical); matrix is authoritative, do not apply another tilt"',
                  text, count=1, flags=re.M)
    text = re.sub(r"^(    measurement_notes:).*$",
                  rf'\g<1> "set_camera_mount.py {when}, lens {fwd * 100:.1f} cm forward, {down * 100:.1f} cm down, '
                  rf'{pitch:g} deg below horizontal (measured on the drone)"',
                  text, count=1, flags=re.M)
    return text


def set_config_mount(pitch: float, fwd: float, down: float, when: str) -> None:
    t = CONFIG.read_text(encoding="utf-8")
    note = f"# set_camera_mount.py {when}"
    for key, val in (("pitch_deg", f"{pitch:.1f}"), ("offset_fwd_m", f"{fwd:.3f}"), ("offset_down_m", f"{down:.3f}")):
        t, n = re.subn(rf"^(    {key}:)\s*[-0-9.]+.*$", rf"\g<1> {val}   {note}", t, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"{CONFIG}: camera.mount.{key} not found - mapper file written, config.yaml unchanged")
    CONFIG.write_text(t, encoding="utf-8")


def show() -> None:
    sys.path.insert(0, str(HERE))
    from miniproject.settings import mount_from_mapper
    src = CURRENT if CURRENT.exists() else BASE
    m = mount_from_mapper(src)
    print(f"runs use {src.name}: lens {m.offset_fwd_m * 100:.1f} cm ahead, {m.offset_down_m * 100:.1f} cm below, "
          f"{m.pitch_deg:.1f} deg below horizontal ({90 - m.pitch_deg:.1f} deg from vertical)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pitch", type=float, help="lens angle below horizontal (deg), measured on the drone")
    ap.add_argument("--fwd", type=float, default=0.095, help="lens ahead of the PX4 (m), default 0.095")
    ap.add_argument("--down", type=float, default=0.025, help="lens below the PX4 (m), default 0.025")
    ap.add_argument("--show", action="store_true", help="print the current mount and stop")
    ap.add_argument("--reset", action="store_true", help="go back to the 29 Oct mount (removes the current file)")
    a = ap.parse_args()
    if a.show:
        show()
        return 0
    if a.reset:
        if CURRENT.exists():
            CURRENT.rename(CURRENT.with_name(f"mapper_config_replaced_{dt.datetime.now():%Y%m%d_%H%M%S}.yaml"))
        set_config_mount(65.0, 0.075, 0.025, f"{dt.date.today()} (reset to 29 Oct)")
        show()
        return 0
    if a.pitch is None:
        ap.error("--pitch is required (lens angle below horizontal, e.g. --pitch 50)")
    if not 20.0 <= a.pitch <= 90.0:
        ap.error("--pitch is degrees BELOW HORIZONTAL, 20..90 (the 29 Oct mount was 65)")
    if not (0.0 <= a.fwd <= 0.3 and -0.1 <= a.down <= 0.2):
        ap.error("--fwd 0..0.3 m and --down -0.1..0.2 m (values are metres, e.g. 0.095)")
    when = f"{dt.datetime.now():%Y-%m-%d %H:%M}"
    mount_id = f"drone5_mount_{dt.datetime.now():%d_%b}_p{a.pitch:g}_f{a.fwd * 100:g}cm"
    text = replace_extrinsic(BASE.read_text(encoding="utf-8"),
                             extrinsic_block(a.pitch, a.fwd, a.down, mount_id, when))
    text = fix_descriptions(text, a.pitch, a.fwd, a.down, when)
    if CURRENT.exists():                                   # keep the previous one, never lose a mount
        CURRENT.rename(CURRENT.with_name(f"mapper_config_replaced_{dt.datetime.now():%Y%m%d_%H%M%S}.yaml"))
    CURRENT.write_text(text, encoding="utf-8")
    set_config_mount(a.pitch, a.fwd, a.down, when)
    show()
    print("every run from now on uses this mount (the first lines of each run print it).  "
          "Fly all experiment flights with the same mount.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

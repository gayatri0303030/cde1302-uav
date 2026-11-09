"""Column-vector transforms T_A_B; tangent order [rotation, translation]."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def so3_exp(v):
    return Rotation.from_rotvec(v).as_matrix()


def so3_log(r):
    return Rotation.from_matrix(r).as_rotvec()


def left_jacobian(v):
    t = np.linalg.norm(v)
    h = skew(v)
    if t < 1e-5:
        return np.eye(3) + (0.5 - t * t / 24) * h + (1 / 6 - t * t / 120) * h @ h
    return np.eye(3) + (1 - np.cos(t)) / t**2 * h + (t - np.sin(t)) / t**3 * h @ h


def transform(r, p):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, p
    return t


def inverse(t):
    r = t[:3, :3].T
    return transform(r, -r @ t[:3, 3])


def se3_exp(x):
    return transform(so3_exp(x[:3]), left_jacobian(x[:3]) @ x[3:])


def se3_log(t):
    v = so3_log(t[:3, :3])
    return np.r_[v, np.linalg.solve(left_jacobian(v), t[:3, 3])]


def adjoint(t):
    r = t[:3, :3]
    return np.block([[r, np.zeros((3, 3))], [skew(t[:3, 3]) @ r, r]])


def log_right_jacobian(x):
    """Inverse right Jacobian via convergent matrix-power integral series."""
    ad = np.block([[skew(x[:3]), np.zeros((3, 3))], [skew(x[3:]), skew(x[:3])]])
    # Jr(x) = integral_0^1 exp(-s ad_x) ds. No SE(3) branch approximation.
    term = np.eye(6)
    jr = term.copy()
    for n in range(1, 60):
        term = term @ (-ad) / (n + 1)
        jr += term
        if np.linalg.norm(term) < 1e-15:
            break
    return np.linalg.solve(jr, np.eye(6))


def pose_from_quaternion(position, xyzw):
    q = np.asarray(xyzw, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 0.01:
        raise ValueError("invalid quaternion (expected normalized xyzw)")
    p = np.asarray(position, float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("invalid 3D position")
    return transform(Rotation.from_quat(q / np.linalg.norm(q)).as_matrix(), p)


def mount(c):
    e = c["camera"]["extrinsic_T_B_C"]
    return transform(np.array(e["rotation_matrix"]), np.array(e["translation_m"]))


def canonical_corners(size):
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])


def ground_corners(tag, size):
    p = canonical_corners(size) @ so3_exp([0, 0, tag[2]]).T
    p[:, :2] += tag[:2]
    return p  # z is identically zero, no state to clamp


def pixel_offset(convention):
    if convention == "apriltag_pixel_edges":
        return 0.5
    if convention == "opencv_pixel_centres":
        return 0.0
    raise ValueError(f"unknown pixel convention {convention!r}")


def project(points, c, convention="opencv_pixel_centres", jacobian=False):
    p = np.asarray(points, float)
    if not np.isfinite(p).all() or np.any(
        p[:, 2] <= c["gating"]["minimum_positive_corner_depth_m"]
    ):
        raise ValueError("corner behind camera or below minimum depth")
    k = np.asarray(c["camera"]["K"])
    k1, k2, p1, p2, k3 = c["camera"]["distortion"]
    x, y = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    r2 = x * x + y * y
    a = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2**3
    xd = x * a + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * a + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    uv = np.c_[k[0, 0] * xd + k[0, 2], k[1, 1] * yd + k[1, 2]] + pixel_offset(
        convention
    )
    if not jacobian:
        return uv
    out = []
    for i in range(len(p)):
        xx, yy, z = x[i], y[i], p[i, 2]
        da = k1 + 2 * k2 * r2[i] + 3 * k3 * r2[i] ** 2
        jdist = np.array(
            [
                [
                    a[i] + 2 * xx * xx * da + 2 * p1 * yy + 6 * p2 * xx,
                    2 * xx * yy * da + 2 * p1 * xx + 2 * p2 * yy,
                ],
                [
                    2 * xx * yy * da + 2 * p1 * xx + 2 * p2 * yy,
                    a[i] + 2 * yy * yy * da + 6 * p1 * yy + 2 * p2 * xx,
                ],
            ]
        )
        out.append(
            np.diag([k[0, 0], k[1, 1]])
            @ jdist
            @ np.array([[1, 0, -xx], [0, 1, -yy]])
            / z
        )
    return uv, np.array(out)


def ground_intersections(pixels, pose, c, convention="opencv_pixel_centres"):
    uv = np.asarray(pixels, float) - pixel_offset(convention)
    if uv.shape != (4, 2) or not np.isfinite(uv).all():
        raise ValueError("expected four finite original corners")
    normalized = cv2.undistortPoints(
        uv.reshape(-1, 1, 2),
        np.array(c["camera"]["K"]),
        np.array(c["camera"]["distortion"]),
    ).reshape(4, 2)
    camera = pose @ mount(c)
    rays = np.c_[normalized, np.ones(4)] @ camera[:3, :3].T
    down = rays[:, 2] / np.linalg.norm(rays, axis=1)
    if np.any(down < c["initialization"]["minimum_unit_ray_down_component"]):
        raise ValueError("ground ray near horizon or points upward")
    distance = -camera[2, 3] / rays[:, 2]
    if np.any(distance <= 0):
        raise ValueError("ground intersection behind camera")
    # Only compute horizontal components. Z=0 is the defining plane equation.
    points = np.zeros((4, 3))
    points[:, :2] = camera[:2, 3] + distance[:, None] * rays[:, :2]
    return points


def initialize_tag(points, size):
    centre = np.mean(points[:, :2], axis=0)
    a, b = canonical_corners(size)[:, :2], points[:, :2] - centre
    yaw = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b))
    return np.r_[centre, yaw]


def pixel_factor(pose, tag, size, pixels, c, convention):
    e = mount(c)
    world = ground_corners(tag, size)
    body = (world - pose[:3, 3]) @ pose[:3, :3]
    camera = (body - e[:3, 3]) @ e[:3, :3]
    uv, jp = project(camera, c, convention, True)
    jpose, jtag = [], []
    for i in range(4):
        jpose.append(jp[i] @ e[:3, :3].T @ np.c_[skew(body[i]), -np.eye(3)])
        delta = world[i] - [tag[0], tag[1], 0]
        jworld = np.array([[1, 0, -delta[1]], [0, 1, delta[0]], [0, 0, 0]])
        jtag.append(jp[i] @ e[:3, :3].T @ pose[:3, :3].T @ jworld)
    return (uv - pixels).reshape(-1), np.vstack(jpose), np.vstack(jtag)

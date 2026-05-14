import base64

import cv2
import numpy as np
from pyquaternion import Quaternion


class PanoramaStitcher:
    """Cylindrical panorama stitcher with precomputed remap tables."""

    def __init__(
        self,
        calibrations: list[dict],
        center_yaw: float,
        h_fov: float = np.pi,
        v_fov: float = 0.65,
        mirror: bool = False,
        width: int = 2400,
        height: int = 480,
    ):
        self.width = width
        self.height = height

        if mirror:
            alpha = np.linspace(-h_fov / 2, h_fov / 2, width)
        else:
            alpha = np.linspace(h_fov / 2, -h_fov / 2, width)
        beta = np.linspace(v_fov / 2, -v_fov / 2, height)
        aa, bb = np.meshgrid(alpha, beta)

        local_x = np.cos(bb) * np.cos(aa)
        local_y = np.cos(bb) * np.sin(aa)
        local_z = np.sin(bb)

        cos_c, sin_c = np.cos(center_yaw), np.sin(center_yaw)
        rays = np.stack(
            [
                cos_c * local_x - sin_c * local_y,
                sin_c * local_x + cos_c * local_y,
                local_z,
            ],
            axis=-1,
        )

        self._cam_maps: list[tuple[np.ndarray, np.ndarray]] = []
        self._cam_weights: list[np.ndarray] = []

        for cal in calibrations:
            K = np.array(cal["intrinsic"])
            R = Quaternion(cal["rotation"]).rotation_matrix
            img_w, img_h = cal["width"], cal["height"]

            rays_cam = rays @ R
            z = rays_cam[..., 2]
            front = z > 0.01
            safe_z = np.where(front, z, 1.0)

            px = K[0, 0] * (rays_cam[..., 0] / safe_z) + K[0, 2]
            py = K[1, 1] * (rays_cam[..., 1] / safe_z) + K[1, 2]

            margin = 1
            valid = (
                front
                & (px >= margin)
                & (px < img_w - margin)
                & (py >= margin)
                & (py < img_h - margin)
            )

            cx, cy = img_w / 2.0, img_h / 2.0
            dist = np.maximum(np.abs(px - cx) / cx, np.abs(py - cy) / cy)
            weight = np.where(valid, np.clip(1.0 - dist * 0.7, 0.05, 1.0), 0.0)

            self._cam_maps.append(
                (
                    np.where(valid, px, 0).astype(np.float32),
                    np.where(valid, py, 0).astype(np.float32),
                )
            )
            self._cam_weights.append(weight.astype(np.float32))

    def stitch(self, images: list[np.ndarray]) -> np.ndarray:
        pano = np.zeros((self.height, self.width, 3), dtype=np.float32)
        total_w = np.zeros((self.height, self.width), dtype=np.float32)

        for i, img in enumerate(images):
            mx, my = self._cam_maps[i]
            w = self._cam_weights[i]
            warped = cv2.remap(
                img, mx, my, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            pano += warped.astype(np.float32) * w[..., np.newaxis]
            total_w += w

        mask = total_w > 0
        pano[mask] /= total_w[mask, np.newaxis]
        return pano.astype(np.uint8)


def encode_panorama(img: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

#!/usr/bin/env python3
# Copyright 2021 Valeo Schalter und Sensoren GmbH and contributors
#
# Author: Christian Witt <christian.witt@valeo.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
import typing

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot


def ensure_point_list(points, dim, concatenate=True, crop=True):
    if isinstance(points, list):
        points = np.array(points)
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2

    if crop:
        for test_dim in range(4, dim, -1):
            if points.shape[1] == test_dim:
                new_shape = test_dim - 1
                assert np.array_equal(points[:, new_shape],
                                      np.ones(points.shape[0]))
                points = points[:, 0:new_shape]

    if concatenate and points.shape[1] == (dim - 1):
        points = np.concatenate((np.array(points), np.ones(
            (points.shape[0], 1))),
                                axis=1)

    if points.shape[1] != dim:
        raise AssertionError('points.shape[1] == dim failed ({} != {})'.format(
            points.shape[1], dim))
    return points


class Projection(object):
    def project_3d_to_2d(self, cam_points: np.ndarray, invalid_value=np.nan):
        raise NotImplementedError()

    def project_2d_to_3d(self, lens_points: np.ndarray, norm: np.ndarray):
        raise NotImplementedError()


class CylindricalProjection(Projection):
    def __init__(self, focal_length: typing.Union[float, list]):
        self.focal_length = focal_length if isinstance(
            focal_length, (float, int)) else focal_length[0]

    def project_3d_to_2d(self, cam_points, invalid_value=np.nan):
        camera_points = ensure_point_list(cam_points, dim=3)

        theta = np.arctan2(camera_points.T[0], camera_points.T[2])
        chi = np.sqrt(camera_points.T[0] * camera_points.T[0] +
                      camera_points.T[2] * camera_points.T[2])

        uv = np.zeros((camera_points.shape[0], 2))
        uv.T[0] = self.focal_length * theta
        uv.T[1] = self.focal_length * camera_points.T[1] * np.divide(
            1, chi, where=(chi != 0))
        uv[chi == 0] = invalid_value
        return uv

    def project_2d_to_3d(self, image_points: np.ndarray, norms: np.ndarray):
        image_points = ensure_point_list(image_points, dim=2)
        norms = ensure_point_list(norms, dim=1)

        outs = np.zeros((image_points.shape[0], 3))

        theta = image_points.T[0] / self.focal_length
        theta_mask = (theta < np.pi / 2) & (theta > -np.pi / 2)
        scale = np.divide(
            norms.flat,
            np.sqrt(image_points.T[1] * image_points.T[1] +
                    self.focal_length * self.focal_length))
        outs.T[0] = self.focal_length * np.sin(theta) * scale * theta_mask
        outs.T[1] = image_points.T[1] * scale * theta_mask
        outs.T[2] = self.focal_length * np.cos(theta) * scale * theta_mask
        return outs


class PerspectProjection(object):
    def __init__(self, intrincs: typing.Union[float, list]):
        self.fx = intrincs[0]
        self.fy = intrincs[1]

    def project_3d_to_2d(self, cam_points: np.ndarray, invalid_value=np.nan):
        raise NotImplementedError()

    def project_2d_to_3d(self, lens_points: np.ndarray, norm: np.ndarray):
        lens_points = ensure_point_list(lens_points, dim=2)
        norms = ensure_point_list(norm, dim=1)

        x1 = lens_points[:, 0] / self.fx
        y1 = lens_points[:, 1] / self.fy
        z1 = np.ones(x1.shape[0], dtype=float)

        return np.vstack((x1, y1, z1)).transpose()


class UCMProjection(Projection):
    def __init__(self, intrincs: typing.Union[float, list]):
        # self.cx = intrincs[0]
        # self.cy = intrincs[1]
        self.gammax = intrincs[2]
        self.gammay = intrincs[3]
        self.xi = intrincs[4]
        self.k1 = intrincs[5]
        self.k2 = intrincs[6]
        self.p1 = intrincs[7]
        self.p2 = intrincs[8]
        self.k3 = intrincs[9]

    def project_3d_to_2d(self, cam_points, invalid_value=np.nan):
        camera_points = ensure_point_list(cam_points, dim=3)
        d = np.sqrt(
            np.power(camera_points[:, 0], 2) +
            np.power(camera_points[:, 1], 2) +
            np.power(camera_points[:, 2], 2))
        x = np.divide(camera_points[:, 0], self.xi * d + camera_points[:, 2])
        y = np.divide(camera_points[:, 1], self.xi * d + camera_points[:, 2])
        r2 = np.power(x, 2) + np.power(y, 2)
        r4 = np.power(r2, 2)
        r6 = np.power(r2, 3)
        x1 = x * (1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
                  ) + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * np.power(x, 2))
        y1 = y * (1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6) + self.p1 * (
            r2 + 2 * np.power(y, 2)) + 2 * self.p2 * x * y
        u1 = self.gammax * x1
        v1 = self.gammay * y1

        return np.vstack((u1, v1)).transpose()

    def project_2d_to_3d(self, lens_points: np.ndarray, norms: np.ndarray):
        lens_points = ensure_point_list(lens_points, dim=2)
        norms = ensure_point_list(norms, dim=1)

        u1 = lens_points[0] / self.gammax
        v1 = lens_points[1] / self.gammay
        # TODO: Add undistort

        u2 = np.power(u1, 2)
        v2 = np.power(v1, 2)
        k = np.divide(self.xi + np.sqrt((1 - self.xi * self.xi) * (u2 + v2)),
                      u2 + v2 + 1)
        x = k * u1
        y = k * v1
        z = (k - self.xi) * np.ones(x.shape[0], dtype=float)
        return np.hstack((x, y, z))


class KBProjection(Projection):
    def __init__(self, intrincs: typing.Union[float, list]):
        self.fx = intrincs[0]
        self.fy = intrincs[1]
        self.cx = intrincs[2]
        self.cy = intrincs[3]
        self.k1 = intrincs[4]
        self.k2 = intrincs[5]
        self.k3 = intrincs[6]
        self.k4 = intrincs[7]

    def project_3d_to_2d(self, cam_points, invalid_value=np.nan):
        camera_points = ensure_point_list(cam_points, dim=3)
        a = np.divide(camera_points[:, 0], camera_points[:, 2])
        b = np.divide(camera_points[:, 1], camera_points[:, 2])
        r = np.sqrt(np.power(a, 2) + np.power(b, 2))
        theta = np.arctan2(r, 1)
        theta_d = self._distort(theta)
        x1 = np.divide(theta_d, r) * a
        y1 = np.divide(theta_d, r) * b
        u1 = self.fx * x1
        v1 = self.fy * y1

        return np.vstack((u1, v1)).transpose()

    def _distort(self, theta):
        theta2 = np.power(theta, 2)
        theta4 = np.power(theta2, 2)
        theta6 = np.power(theta2, 3)
        theta8 = np.power(theta2, 4)
        return theta * (1 + self.k1 * theta2 + self.k2 * theta4 +
                        self.k3 * theta6 + self.k4 * theta8)


class RadialPolyCamProjection(Projection):
    def __init__(self, distortion_params: list):
        self.coefficients = np.asarray(distortion_params)
        self.power = np.array(
            [np.arange(start=1, stop=self.coefficients.size + 1)]).T

    def project_3d_to_2d(self, cam_points, invalid_value=np.nan):
        camera_points = ensure_point_list(cam_points, dim=3)
        chi = np.sqrt(camera_points.T[0] * camera_points.T[0] +
                      camera_points.T[1] * camera_points.T[1])
        theta = np.pi / 2.0 - np.arctan2(camera_points.T[2], chi)
        rho = self._theta_to_rho(theta)
        lens_points = np.divide(
            rho, chi, where=(chi != 0))[:, np.newaxis] * camera_points[:, 0:2]

        # set (0, 0, 0) = np.nan
        lens_points[(chi == 0) & (cam_points[:, 2] == 0)] = invalid_value
        return lens_points

    def project_2d_to_3d(self, lens_points: np.ndarray, norms: np.ndarray):
        lens_points = ensure_point_list(lens_points, dim=2)
        norms = ensure_point_list(norms, dim=1).reshape(norms.size)

        rhos = np.linalg.norm(lens_points, axis=1)
        thetas = self._rho_to_theta(rhos)
        chis = norms * np.sin(thetas)
        zs = norms * np.cos(thetas)
        xy = np.divide(chis, rhos, where=(rhos != 0))[:,
                                                      np.newaxis] * lens_points
        xyz = np.hstack((xy, zs[:, np.newaxis]))
        return xyz

    def _theta_to_rho(self, theta):
        return np.dot(self.coefficients, np.power(np.array([theta]),
                                                  self.power))

    def _rho_to_theta(self, rho):
        coeff = list(reversed(self.coefficients))
        results = np.zeros_like(rho)
        for i, _r in enumerate(rho):
            theta = np.roots([*coeff, -_r])
            theta = np.real(theta[theta.imag == 0])
            theta = theta[np.where(np.abs(theta) < np.pi)]
            theta = np.min(theta) if theta.size > 0 else 0
            results[i] = theta
        return results


class RadialTableCamProjection(Projection):
    def __init__(self, distortion_params: list):
        self.coefficients = np.asarray(distortion_params)

        import os
        p = os.path.dirname(__file__)
        file_name = p + "/4066_distortion.xls"
        self._read_table(file_name)
        self.focal_length = distortion_params[0]

    def project_3d_to_2d(self, cam_points, invalid_value=np.nan):
        camera_points = ensure_point_list(cam_points, dim=3)
        lens_points = np.zeros((camera_points.shape[0], 2))
        camera_points_x = np.zeros(camera_points.shape[0], dtype=float)
        camera_points_y = np.zeros(camera_points.shape[0], dtype=float)
        camera_points_x = np.divide(camera_points.T[0],
                                    camera_points.T[2],
                                    where=(camera_points.T[2] != 0))
        camera_points_y = np.divide(camera_points.T[1],
                                    camera_points.T[2],
                                    where=(camera_points.T[2] != 0))
        chi = np.sqrt(camera_points_x * camera_points_x +
                      camera_points_y * camera_points_y)
        theta = np.arctan2(chi, 1)
        theta[chi < 0.00001] = 0
        theta_mask = (theta > np.pi) | (theta < 0)
        theta = np.clip(theta, 0, np.pi)

        dist = self._find_dist_np(theta)
        lens_points[:, 0] = camera_points_x * dist * self.focal_length
        lens_points[:, 1] = camera_points_y * dist * self.focal_length

        # set (0, 0, 0) = np.nan
        lens_points[(chi == 0) | (cam_points[:, 2] == 0)
                    | theta_mask] = invalid_value
        return lens_points

    def _read_table(self, file_name):
        # -----------------------distortion table--------------
        import pandas as pd
        xl_file = pd.ExcelFile(file_name)
        dfs = {
            sheet_name: xl_file.parse(sheet_name)
            for sheet_name in xl_file.sheet_names
        }

        distortion = dfs['distortion'].values
        self.angle_table = distortion[:900, 0] * np.pi / 180
        self.ref_table = distortion[:900, 1]
        self.real_table = distortion[:900, 2]
        self.dist_table = distortion[:900, 3] * 0.01

    def _binary_search(self, arr, l, r, x):
        while l < r:
            mid = (l + r) // 2
            if x > arr[mid]:
                l = mid + 1
            else:
                r = mid
        return r

    def _find_dist(self, theta):
        idx = self._binary_search(self.angle_table, 0,
                                  len(self.angle_table) - 1, theta)
        left_ref = self.angle_table[idx - 1]
        right_ref = self.angle_table[idx]
        ratio = (theta - left_ref) / (right_ref - left_ref)
        left_dist = self.dist_table[idx - 1]
        right_dist = self.dist_table[idx]
        target_dist = (right_dist - left_dist) * ratio + left_dist
        return 1 + target_dist

    def _find_dist_np(self, theta):
        interval = 0.1 * np.pi / 180
        idx_f = theta / interval
        idx_i = np.floor(idx_f).astype(int)
        idx_i = np.clip(idx_i, 0, 899)

        left_ref = np.take(self.angle_table, idx_i)
        ratio = (theta - left_ref) / interval
        left_dist = np.take(self.dist_table, idx_i)
        right_dist = np.take(self.dist_table, np.clip(idx_i + 1, 0, 899))
        target_dist = (right_dist - left_dist) * ratio + left_dist
        return 1 + target_dist


class Camera(object):
    def __init__(self,
                 lens: Projection,
                 translation,
                 rotation,
                 size,
                 principle_point,
                 aspect_ratio: float = 1.0):
        self.lens = lens
        pose = np.eye(4)
        pose[0:3, 3] = translation
        pose[0:3, 0:3] = rotation
        self._pose = np.asarray(pose, dtype=float)
        self._inv_pose = np.linalg.inv(self._pose)
        self._size = np.array([size[0], size[1]], dtype=int)
        self._principle_point = 0.5 * self._size + np.array(
            [principle_point[0], principle_point[1]], dtype=float) - 0.5
        self._aspect_ratio = np.array([1, aspect_ratio], dtype=float)

    size = property(lambda self: self._size)
    width = property(lambda self: self._size[0])
    height = property(lambda self: self._size[1])
    cx = property(lambda self: self._principle_point[0])
    cy = property(lambda self: self._principle_point[1])
    cx_offset = property(
        lambda self: self._principle_point[0] - 0.5 * self._size[0] + 0.5)
    cy_offset = property(
        lambda self: self._principle_point[1] - 0.5 * self._size[1] + 0.5)
    aspect_ratio = property(lambda self: self._aspect_ratio[1])

    rotation = property(lambda self: self._pose[0:3, 0:3])
    translation = property(lambda self: self._pose[0:3, 3])

    def project_3d_to_2d(self,
                         world_points: np.ndarray,
                         do_clip=False,
                         invalid_value=np.nan):
        world_points = ensure_point_list(world_points, dim=4)
        # print(self._pose)
        # print(self._inv_pose)
        camera_points = world_points @ self._inv_pose.T
        lens_points = self.lens.project_3d_to_2d(camera_points[:, 0:3],
                                                 invalid_value=invalid_value)
        screen_points = (lens_points *
                         self._aspect_ratio) + self._principle_point
        return self._apply_clip(screen_points,
                                screen_points) if do_clip else screen_points

    def project_2d_to_3d(self,
                         screen_points: np.ndarray,
                         norm: np.ndarray,
                         do_clip=False):
        screen_points = ensure_point_list(screen_points,
                                          dim=2,
                                          concatenate=False,
                                          crop=False)
        norm = ensure_point_list(norm[:, np.newaxis],
                                 dim=1,
                                 concatenate=False,
                                 crop=False)
        lens_points = (screen_points -
                       self._principle_point) / self._aspect_ratio
        lens_points = self._apply_clip(
            lens_points, screen_points) if do_clip else lens_points

        camera_points = self.lens.project_2d_to_3d(lens_points, norm)

        camera_points = ensure_point_list(camera_points, dim=4)
        world_points = camera_points @ self._pose.T
        return world_points[:, 0:3]

    def _apply_clip(self, points, clip_source) -> np.ndarray:
        if self._size[0] == 0 or self._size[1] == 0:
            raise RuntimeError('clipping without a size is not possible')
        mask = (clip_source[:, 0] < 0) | (clip_source[:, 0] >= self._size[0]) | \
               (clip_source[:, 1] < 0) | (clip_source[:, 1] >= self._size[1])

        points[mask] = [np.nan]
        return points


def create_img_projection_maps(source_cam: Camera, destination_cam: Camera):
    """generates maps for cv2.remap to remap from one camera to another"""
    u_map = np.zeros((destination_cam.height, destination_cam.width, 1),
                     dtype=np.float32)
    v_map = np.zeros((destination_cam.height, destination_cam.width, 1),
                     dtype=np.float32)

    destination_points_b = np.arange(destination_cam.height)

    for u_px in range(destination_cam.width):
        destination_points_a = np.ones(destination_cam.height) * u_px
        destination_points = np.vstack(
            (destination_points_a, destination_points_b)).T

        source_points = source_cam.project_3d_to_2d(
            destination_cam.project_2d_to_3d(destination_points,
                                             norm=np.array([1])))

        u_map.T[0][u_px] = source_points.T[0]
        v_map.T[0][u_px] = source_points.T[1]

    map1, map2 = cv2.convertMaps(u_map,
                                 v_map,
                                 dstmap1type=cv2.CV_16SC2,
                                 nninterpolation=False)
    return map1, map2


def create_ipm_projection_maps(dst_box: np.ndarray, cam: Camera):
    """generates maps for cv2.remap to remap from one camera to another"""
    # dst: top, left, bottom, right: 9, 6, 4, -6
    y_axis = np.arange(dst_box[0], dst_box[2], -0.02)  # 9, 4
    x_axis = np.arange(dst_box[1], dst_box[3], -0.02)  # 6, -6
    u_map = np.zeros((y_axis.shape[0], x_axis.shape[0], 1), dtype=np.float32)
    v_map = np.zeros((y_axis.shape[0], x_axis.shape[0], 1), dtype=np.float32)
    for u, px in enumerate(x_axis):
        destination_points_x = np.ones(y_axis.shape[0]) * px
        destination_points_z = np.ones(y_axis.shape[0]) * 0.0
        destination_points = np.vstack(
            (y_axis, destination_points_x, destination_points_z)).T

        source_points = cam.project_3d_to_2d(destination_points)
        u_map.T[0][u] = source_points.T[0]
        v_map.T[0][u] = source_points.T[1]

    map1, map2 = cv2.convertMaps(u_map,
                                 v_map,
                                 dstmap1type=cv2.CV_16SC2,
                                 nninterpolation=False)
    return map1, map2


def read_cam_from_json(path):
    """generates a Camera object from a json file"""
    with open(path) as f:
        config = json.load(f)

    intrinsic = config['intrinsic']
    coefficients = [
        intrinsic['k1'], intrinsic['k2'], intrinsic['k3'], intrinsic['k4']
    ]

    cam = Camera(rotation=SciRot.from_quat(
        config['extrinsic']['quaternion']).as_matrix(),
                 translation=config['extrinsic']['translation'],
                 lens=RadialPolyCamProjection(coefficients),
                 size=(intrinsic['width'], intrinsic['height']),
                 principle_point=(intrinsic['cx_offset'],
                                  intrinsic['cy_offset']),
                 aspect_ratio=intrinsic['aspect_ratio'])

    return cam

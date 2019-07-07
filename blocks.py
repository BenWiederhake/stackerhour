#!/usr/bin/env python3

import math
import random

import cairo

# == ABOUT: MODEL ==
# All angles are in radians, i.e., in [0, 2pi).
# All blocks have size 1×1×1.
# Coordinate system is:
#   y↑
#    |
#  ——+—→
#    | x
# 90° rotation usually maps (x,y) to (-y,x).
# The 3D coordinate system is right-handed.

# == CONFIG: GENERATION ==
N_BLOCKS = 20
BLOCK_POS_VARIANCE = math.sqrt(N_BLOCKS) / 2

# == CONFIG: RENDERING ==
CAMERA_POS = (-3, -10, 3)
IMAGE_RES = (800, 600)
# Length of 1 unit on projection plane (which has distance 1 from the camera)
# in terms of pixels on the image:
CAMERA_SCALE = 500
CAMERA_LOOKAT = (0, 0, 1)


# == CODE: MATH ==

def vec_sqd(dxyz):
    dx, dy, dz = dxyz
    return dx * dx + dy * dy + dz * dz


def vec_scale(dxyz, factor):
    dx, dy, dz = dxyz
    return dx * factor, dy * factor, dz * factor


def vec_normalize(dxyz):
    return vec_scale(dxyz, 1 / math.sqrt(vec_sqd(dxyz)))


def vec_sub(dxyz1, dxyz2):
    x1, y1, z1 = dxyz1
    x2, y2, z2 = dxyz2
    return x1 - x2, y1 - y2, z1 - z2


def vec_scalar(dxyz1, dxyz2):
    x1, y1, z1 = dxyz1
    x2, y2, z2 = dxyz2
    return x1 * x2 + y1 * y2 + z1 * z2


def vec_cross(dxyz1, dxyz2):
    x1, y1, z1 = dxyz1
    x2, y2, z2 = dxyz2
    return y1 * z2 - y2 * z1, z1 * x2 - z2 * x1, x1 * y2 - x2 * y1


CAMERA_FRONT = vec_normalize(vec_sub(CAMERA_LOOKAT, CAMERA_POS))
CAMERA_UP_PRE = (0, 1, 0)
CAMERA_RIGHT = vec_normalize(vec_cross(CAMERA_FRONT, CAMERA_UP_PRE))
# assert vec_sqd(vec_cross(CAMERA_FRONT, CAMERA_UP_PRE)) >= 0.01
# Violated on sharp up or down perspectives.
CAMERA_UP = vec_normalize(vec_cross(CAMERA_RIGHT, CAMERA_FRONT))


def vec_project(dxyz):
    dxyz = vec_sub(dxyz, CAMERA_POS)
    img_x = vec_scalar(dxyz, CAMERA_RIGHT)
    img_y = vec_scalar(dxyz, CAMERA_UP)
    img_z = vec_scalar(dxyz, CAMERA_FRONT)
    # … also known as "matrix multiplication".  I know.
    return img_x, img_y, img_z


# == CODE: BUSINESS ==

class Block:
    def __init__(self, x, y, a, z=0):
        self.x = x
        self.y = y
        self.a = a
        self.ex, self.ey = math.cos(self.a) * 0.5, math.sin(self.a) * 0.5
        self.cx, self.cy = self.ex - self.ey, self.ey + self.ex
        self.z = z

    def is_intersecting(self, other):
        assert self.z == other.z, 'Caller needs to ensure that'
        sqdist = (self.x - other.x) ** 2 + (self.y + other.y) ** 2
        if sqdist <= 1:
            # Minimum diameter is 1 each, so they *must* intersect.
            return True
        if sqdist > 2:
            # Maximum diameter is sqrt(2) each, so they *cannot* intersect.
            return False

        # Oh well, now it's a bit more difficult.
        # We have sqdist > 1, which tells us that *neither* center point
        # is in the other block.  I can't prove it,  # FIXME
        # but I think this means that "intersection IFF any corner is
        # contained in the other".
        # Before we checked the distances, this was not true: The squares could
        # be for example in the same spot, rotated 0° and 45° respectively.
        return any(self.contains(c) for c in other.corners) \
            or any(other.contains(c) for c in self.corners)

    def contains(self, p):
        px, py = p
        px, py = px - self.x, py - self.y
        e1_dist = self.ex * px + self.ey * py
        e2_dist = -self.ey * px + self.ex * py
        # Edges are only 0.5 away, and we need to normalize by
        # the length of the vector (ex, ey), which is 0.5.
        return -0.25 <= e1_dist <= 0.25 and -0.25 <= e2_dist <= 0.25

    @property
    def corners(self):
        u, v = self.cx, self.cy
        x, y = self.x, self.y
        return [(x + u, y + v), (x - v, y + u), (x - u, y - u), (x + v, y - u)]

    @staticmethod
    def new_random():
        return Block(random.gauss(0, BLOCK_POS_VARIANCE), random.gauss(0, BLOCK_POS_VARIANCE),
                     random.random() * 2 * math.pi)

    def __repr__(self):
        return 'Block(x={}, y={}, a={}, z={})'.format(self.x, self.y, self.a, self.z)


def compute_blocks():
    blocks_by_level = []
    for _ in range(N_BLOCKS):
        b = Block.new_random()
        while True:
            if len(blocks_by_level) <= b.z:
                blocks_by_level.append([b])
                # It's the only block on that level → Done!
                break
            if not any(lb.is_intersecting(b) for lb in blocks_by_level[b.z]):
                # It doesn't intersect anymore → Done!
                blocks_by_level[b.z].append(b)
                break
            # It intersects with something → Go one level up.
            b.z += 1
    return [b for l in blocks_by_level for b in l]


def render_blocks(blocks):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, *IMAGE_RES)
    ctx = cairo.Context(surface)
    ctx.scale(*IMAGE_RES)
    raise NotImplementedError()


def run():
    blocks = compute_blocks()
    img = render_blocks(blocks)
    img.save('blocks.png')


if __name__ == '__main__':
    run()

#!/usr/bin/env python3

import math
import random

from PIL.Image import Image

# All angles are in radians, i.e., in [0, 2pi).
# All blocks have size 1×1×1.
# Coordinate system is:
#   y↑
#    |
#  ——+—→
#    | x
# 90° rotation usually maps (x,y) to (-y,x).


BLOCK_POS_VARIANCE = 3.


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
    raise NotImplementedError()


def render_blocks(blocks):
    raise NotImplementedError()


def run():
    blocks = compute_blocks()
    img = render_blocks(blocks)
    img.save('blocks.png')


if __name__ == '__main__':
    run()

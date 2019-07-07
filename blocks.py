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
N_BLOCKS = 100
BLOCK_POS_SIGMA = 7

# == CONFIG: RENDERING ==
CAMERA_POS = (-2, -20, 6)
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1024
BLOCK_RGB_MU = 0.5
BLOCK_RGB_SIGMA = 0.19
NORMAL_RGB_IMPACT = 0.1
# Length of 1 unit on projection plane (which has distance 1 from the camera)
# in terms of pixels on the image:
CAMERA_SCALE = 1000
CAMERA_LOOKAT = (0, 0, 0)

BACKGROUND_X = 31
BACKGROUND_Y = 31
BACKGROUND_STEP = 1
BACKGROUND_W = 0.95
BACKGROUND_H = 0.95
BACKGROUND_Z = -0.01
BACKGROUND_META = (0.85, 0.86, 0.83)


# == CODE: MATH ==

def vec_sqd(dxyz):
    dx, dy, dz = dxyz
    return dx * dx + dy * dy + dz * dz


def vec_scale(dxyz, factor):
    dx, dy, dz = dxyz
    return dx * factor, dy * factor, dz * factor


def vec_normalize(dxyz):
    return vec_scale(dxyz, 1 / math.sqrt(vec_sqd(dxyz)))


def vec_add(dxyz1, dxyz2):
    x1, y1, z1 = dxyz1
    x2, y2, z2 = dxyz2
    return x1 + x2, y1 + y2, z1 + z2


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
CAMERA_UP_PRE = (0, 0, 1)
CAMERA_RIGHT = vec_normalize(vec_cross(CAMERA_FRONT, CAMERA_UP_PRE))
CAMERA_UP = vec_normalize(vec_cross(CAMERA_RIGHT, CAMERA_FRONT))
CAMERA_CUTOFF = 0.7501  # Minimal distance at which things need to be.

print(f'· {CAMERA_FRONT}')
print(f'→ {CAMERA_RIGHT}')
print(f'↑ {CAMERA_UP}')
assert 0.999 < vec_sqd(CAMERA_FRONT) < 1.001
assert 0.999 < vec_sqd(CAMERA_RIGHT) < 1.001
assert 0.999 < vec_sqd(CAMERA_UP) < 1.001
assert -0.001 < vec_scalar(CAMERA_FRONT, CAMERA_RIGHT) < 0.001
assert -0.001 < vec_scalar(CAMERA_UP, CAMERA_RIGHT) < 0.001
assert -0.001 < vec_scalar(CAMERA_FRONT, CAMERA_UP) < 0.001


def vec_project(dxyz):
    dxyz = vec_sub(dxyz, CAMERA_POS)
    img_z = vec_scalar(dxyz, CAMERA_FRONT)
    img_x = vec_scalar(dxyz, CAMERA_RIGHT) / img_z
    img_y = -vec_scalar(dxyz, CAMERA_UP) / img_z
    # … also known as "matrix multiplication in homomorphic coordinates".  I know.
    return img_x, img_y, img_z


def clamp_rgbish(val):
    return max(0.0, min(val * 1.0, 1.0))


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
        sqdist = (self.x - other.x) ** 2 + (self.y - other.y) ** 2
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
        return any(self.contains(c) for c in other.corners()) \
            or any(other.contains(c) for c in self.corners())

    def contains(self, p):
        px, py = p
        px, py = px - self.x, py - self.y
        e1_dist = self.ex * px + self.ey * py
        e2_dist = -self.ey * px + self.ex * py
        # Edges are only 0.5 away, and we need to normalize by
        # the length of the vector (ex, ey), which is 0.5.
        return -0.25 <= e1_dist <= 0.25 and -0.25 <= e2_dist <= 0.25

    def corners(self):
        u, v = self.cx, self.cy
        x, y = self.x, self.y
        return [(x + u, y + v), (x - v, y + u), (x - u, y - v), (x + v, y - u)]

    @staticmethod
    def new_random():
        return Block(random.gauss(0, BLOCK_POS_SIGMA), random.gauss(0, BLOCK_POS_SIGMA),
                     random.random() * 2 * math.pi)

    def __repr__(self):
        return 'Block(x={}, y={}, a={}, z={})'.format(self.x, self.y, self.a, self.z)


def block_to_rgb(b, rect_normal):
    r = random.Random(b)
    garg = (BLOCK_RGB_MU, BLOCK_RGB_SIGMA)
    # Warm up rng, just in case.
    r.gauss(*garg)
    r.gauss(*garg)
    return [clamp_rgbish(r.gauss(*garg) + rect_normal[i] * NORMAL_RGB_IMPACT)
            for i in range(3)]


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


def compute_rects(blocks):
    '''
    Yields rectangles whose outside face is counter-clockwise.
    Specifically, yields (rectangle, block) tuples.
    '''
    rects = []
    for b in blocks:
        plane_distance = vec_project((b.x, b.y, b.z + 0.5))[2]
        if plane_distance <= CAMERA_CUTOFF:
            # Don't render this block, it's too close to the camera.
            continue

        c1b, c2b, c3b, c4b = [(cx, cy, b.z) for cx, cy in b.corners()]
        c1t, c2t, c3t, c4t = [(cx, cy, b.z + 1) for cx, cy in b.corners()]
        # Bottom
        rects.append(([c1b, c2b, c3b, c4b], b))
        # Top
        rects.append(([c4t, c3t, c2t, c1t], b))
        # Sides
        rects.append(([c1t, c4t, c4b, c1b], b))
        rects.append(([c2t, c1t, c1b, c2b], b))
        rects.append(([c3t, c2t, c2b, c3b], b))
        rects.append(([c4t, c3t, c3b, c4b], b))
    return rects


def project_rect(rect):
    '''
    Projects a rect onto the projection plane, and gives
    the depth at the actual center of the rectangle.
    '''
    a, _, c, _ = rect
    center = vec_scale(vec_add(a, c), 0.5)
    projected = [vec_project(v)[:2] for v in rect]
    return (projected, vec_project(center)[2])


def compute_paint_order(blocks):
    paint_rects = []
    for rect, b in compute_rects(blocks):
        b_depth = distance = vec_project((b.x, b.y, b.z + 0.5))[2]  # TODO: Duplicate computation!
        projected, rect_depth = project_rect(rect)
        rect_normal = vec_cross(vec_sub(rect[1], rect[0]), vec_sub(rect[2], rect[1]))
        assert 0.99 <= vec_sqd(rect_normal) <= 1.01, (rect_normal, vec_sqd(rect_normal), rect, b)
        meta = block_to_rgb(b, rect_normal)
        paint_rects.append((b_depth, rect_depth, projected, meta))
    for bg_x in range(BACKGROUND_X):
        for bg_y in range(BACKGROUND_Y):
            base_x = (bg_x - (BACKGROUND_X - 0.5) / 2) * BACKGROUND_STEP
            base_y = (bg_y - (BACKGROUND_Y - 0.5) / 2) * BACKGROUND_STEP
            off_x = BACKGROUND_W / 2
            off_y = BACKGROUND_H / 2
            rect = [(base_x + off_x, base_y + off_y, BACKGROUND_Z),
                    (base_x - off_x, base_y + off_y, BACKGROUND_Z),
                    (base_x - off_x, base_y - off_y, BACKGROUND_Z),
                    (base_x + off_x, base_y - off_y, BACKGROUND_Z)]
            projected, rect_depth = project_rect(rect)
            paint_rects.append((rect_depth, rect_depth, projected, BACKGROUND_META))
    paint_rects.sort(reverse=True)
    paint_rects = [e[2:] for e in paint_rects]
    return paint_rects


def render_blocks(blocks):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, IMAGE_WIDTH, IMAGE_HEIGHT)
    ctx = cairo.Context(surface)
    ctx.translate(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2)
    ctx.scale(CAMERA_SCALE, CAMERA_SCALE)

    for rect, rgb in compute_paint_order(blocks):
        ctx.move_to(*rect[0])
        ctx.line_to(*rect[1])
        ctx.line_to(*rect[2])
        ctx.line_to(*rect[3])
        ctx.close_path()
        ctx.set_source_rgb(*rgb)
        ctx.set_line_width(0.02)
        ctx.fill()

    return surface


def run():
    blocks = compute_blocks()
    # a = 45 * math.pi / 180
    # blocks = [
    #     Block(-4, -4, a), Block(-2, -4, a), Block(0, -4, a), Block(2, -4, a), Block(4, -4, a),
    #     Block(-4, -2, a), Block(-2, -2, a), Block(0, -2, a), Block(2, -2, a), Block(4, -2, a),
    #     Block(-4,  0, a), Block(-2,  0, a), Block(0,  0, a), Block(2,  0, a), Block(4,  0, a),
    #     Block(-4,  2, a), Block(-2,  2, a), Block(0,  2, a), Block(2,  2, a), Block(4,  2, a),
    #     Block(-4,  4, a), Block(-2,  4, a), Block(0,  4, a), Block(2,  4, a), Block(4,  4, a),
    #     Block(0, -20, a, z=4),
    # ]
    surface = render_blocks(blocks)
    surface.write_to_png("stackerhour.png")


if __name__ == '__main__':
    run()

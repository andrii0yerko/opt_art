from PIL import Image, ImageDraw, ImageOps
from functools import lru_cache
import numpy as np


from .utils import join_images


class ElasticTruchetTiles:
    def __init__(self):
        pass

    @classmethod
    @lru_cache(1024)
    def _get_tile(cls, template="c", t=0.5, side=20):
        # TODO: lazy creation of tiles
        m = side * 0.25
        M = side * 0.75
        pos = (M - m) * (1 - t) + m

        out = Image.new("RGB", (side, side), (255, 255, 255))
        d = ImageDraw.Draw(out)

        xy = [(0, 0), (side - pos, pos), (side, side), (side, 0)]
        d.polygon(xy, fill=(0, 0, 0), outline=None)

        if template == "c":
            pass
        elif template == "a":
            out = out.rotate(180)
        elif template == "b":
            out = out.rotate(90)
        elif template == "d":
            out = out.rotate(-90)
        return out

    @classmethod
    def _find_optimum(cls, b):
        if b < 0.25:
            return 0
        if b > 0.75:
            return 1
        return 2 * b - 0.5

    @staticmethod
    def _resolve_pattern(pattern, shape):
        if pattern == "random":
            return np.random.choice(["a", "b", "c", "d"], shape)

        if isinstance(pattern, str):
            assert not set(pattern) - set("abcd "), ValueError("Pattern must consist only of {'a', 'b', 'c', 'd', ' '}")
            pattern = list(map(list, pattern.split(" ")))

        pattern = np.array(pattern)

        reps = np.ceil(np.divide(shape, pattern.shape)).astype(int)
        checkerboard = np.tile(pattern, reps)
        checkerboard = checkerboard[: shape[0], : shape[1]]
        return checkerboard

    @staticmethod
    def _preprocess_image(img, block_size):
        new_shape = np.floor_divide(img.size, (block_size, block_size))
        img = img.resize(new_shape)
        img = np.array(ImageOps.grayscale(img))
        img = np.array(img) / 255
        return img

    def transform(
        self,
        img,
        block_size,
        pattern="ac ca",
    ):

        original_size = img.size
        img = self._preprocess_image(img, block_size)
        checkerboard = self._resolve_pattern(pattern, img.shape)
        tiles = [self._get_tile(template, self._find_optimum(window)) for window, template in zip(img.ravel(), checkerboard.ravel())]

        tiles = [tiles[i * img.shape[1]: (i + 1) * img.shape[1]] for i in range(img.shape[0])]

        return join_images(*tiles).resize(original_size)

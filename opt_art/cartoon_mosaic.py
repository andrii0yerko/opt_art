from typing import Sequence

import numpy as np
from PIL import Image
from pulp import LpMinimize, LpProblem, LpVariable, LpSolverDefault

from .utils import join_images


# TODO allow RGB images
# TODO: investigate possibilities use
#   - nonsquare tiles
#   - pass in num_copies instead of block size
#   inspiration: https://youtu.be/vXWvptwoCl8
class CartoonMosaics:
    def __init__(self, images: Sequence[Image.Image], pulp_solver=None):
        self._images = [x.convert("L") for x in images]
        if not self._images:
            raise ValueError("Input image sequence is empty")
        if not pulp_solver:
            pulp_solver = type(LpSolverDefault)(msg=0)
        self.pulp_solver = pulp_solver

    def _find_optimum(self, processed_image):
        max_size = processed_image.shape[1:]
        set_len = len(self._images)
        brightnesses = np.array([np.mean(x) for x in self._images]).reshape(-1, 1, 1) / 255
        num_copies = np.prod(max_size) // set_len

        vars_shape = (set_len, *max_size)
        idx = list(np.ndindex(vars_shape))

        prob = LpProblem("opt_art.CartoonMosaics", LpMinimize)

        # define variables
        loc_vars_dict = LpVariable.dicts("loc", idx, lowBound=0, upBound=1, cat="Integer")
        loc_vars = list(loc_vars_dict[x] for x in idx)
        loc_vars = np.array(loc_vars).reshape(vars_shape)

        # objective
        prob += (loc_vars * (brightnesses - processed_image) ** 2).sum()

        # each mosaic image must appear exactly num_copies times
        for layer in loc_vars:
            prob += layer.sum() == num_copies

        # only one image on each pixel
        for i, j in np.ndindex(*max_size):
            prob += loc_vars[:, i, j].sum() == 1

        # TODO: benchmark different optimizers
        prob.solve(self.pulp_solver)

        result = np.vectorize(lambda x: x.value())(loc_vars)
        result = result.argmax(axis=0)
        return result

    def _preprocess_image(self, img, block_size):
        set_len = len(self._images)

        max_size = np.floor_divide(img.size, (block_size, block_size))
        max_size = np.floor_divide(max_size, set_len) * set_len

        processed_image = img.convert("L").resize(max_size)
        processed_image = np.array(processed_image)
        processed_image = processed_image.reshape(1, *processed_image.shape) / 255

        return processed_image

    def transform(self, img, approx_block_size):
        original_size = img.size
        img = self._preprocess_image(img, approx_block_size)
        tiles = self._find_optimum(img)

        tiles = [[self._images[y] for y in x] for x in tiles]

        return join_images(*tiles).resize(original_size)

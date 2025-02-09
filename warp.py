import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import swirl

def check_bounds(a, mx):
    msg = 'dimension {} of coordinate array has entries outside [0, {})'
    for d in (0, 1):
        assert np.all((0 <= a[d]) & (a[d] <= mx[d] - 1)), msg.format(d, mx[d])

def warp(in_image, coords):
    assert in_image.ndim == 2, 'input image must be gray'
    assert coords.ndim == 3, 'coordinates array must be 3D'
    assert coords.shape[0] == 2, 'coordinates array must be 2 x rows x cols'

    in_shape = in_image.shape
    check_bounds(coords, in_shape)

    x = coords[0] # a
    y = coords[1] # b

    coords_floored = np.floor(coords).astype(int) # floor a and floor b
    floor_x = coords_floored[0] # floor a
    floor_y = coords_floored[1] # floor b

    floor_x = np.clip(floor_x, 0, in_shape[0] - 2)
    floor_y = np.clip(floor_y, 0, in_shape[1] - 2)

    deltas = coords - coords_floored # delta a and delta b
    delta_x = (coords - coords_floored)[0] # delta a
    delta_y = (coords - coords_floored)[1] # delta b

    out_image = np.empty(shape=(coords.shape[1], coords.shape[2]))

    sum1 = in_image[floor_x, floor_y] * (1 - delta_x) * (1 - delta_y)
    sum2 = in_image[floor_x, floor_y + 1] * (1 - delta_x) * delta_y
    sum3 = in_image[floor_x + 1, floor_y] * delta_x * (1 - delta_y)
    sum4 = in_image[floor_x + 1, floor_y + 1] * delta_x * delta_y

    out_image = sum1 + sum2 + sum3 + sum4

    return out_image


def sample(image, rows, columns):
    a, b = np.meshgrid(rows, columns, indexing='ij')
    coords = np.array([a, b])
    
    out_image = warp(image, coords)
    return out_image


def swirl(image, center=None, strength=10, radius=120, rotation=0):
    if center is None:
        center = np.array(image.shape) / 2

    rows, cols = np.arange(image.shape[0]), np.arange(image.shape[1])
    a, b = np.meshgrid(rows, cols, indexing='ij')

    centered_a = a - center[0]
    centered_b = b - center[1]

    rho = np.sqrt(centered_a ** 2 + centered_b ** 2)
    theta = np.arctan2(centered_b, centered_a)  

    adjusted_radius = radius / (5 * np.log(2))

    old_theta = -rotation - (strength * np.exp(-rho / adjusted_radius)) + theta

    rows1 = center[0] + rho * np.cos(old_theta)
    columns1 = center[1] + rho * np.sin(old_theta)

    rows1 = np.clip(rows1, 0, image.shape[0] - 1) # can disregard
    columns1 = np.clip(columns1, 0, image.shape[1] - 1) # can disregard

    coords = np.array([rows1, columns1])
    out_image = warp(image, coords)

    return out_image


if __name__ == "__main__":
    I = 10 * np.reshape(np.arange(6), (2, 3))
    row_grid = np.linspace(0, 1, 5)
    column_grid = np.linspace(1, 2, 3)

    J = sample(I, row_grid, column_grid)
    print(J)

    #-------------------------------------------------------

    from skimage import data

    page = data.page()
    page_detail = sample(page, np.linspace(47, 65, 100), np.linspace(88, 160, 200))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(page, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(page_detail, cmap='gray')
    plt.axis('off')

    plt.show()

    #-------------------------------------------------------

    checkerboard = data.checkerboard()
    swirled_image = swirl(checkerboard)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(checkerboard, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(swirled_image, cmap='gray')
    plt.axis('off')

    plt.show()
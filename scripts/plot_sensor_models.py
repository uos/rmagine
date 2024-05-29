
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sensor_models import deg2rad
from sensor_models import SphericalModel, PinholeModel


"""
Tested with 
- ubuntu20, python 3.8.10, numpy 1.24.3, matplotlib 3.1.2
"""


"""
Generate ray directions from model
"""
def gen_ray_dirs(model, norm_x = False):
    ray_dirs = []
    for vid in range(model.getHeight()):
        for hid in range(model.getWidth()):
            dir = model.getDirection(vid, hid)
            if norm_x:
                dir /= dir[0]
            ray_dirs.append(dir)

    ray_dirs = np.array(ray_dirs).T
    return ray_dirs


def my_spherical_model():
    model = SphericalModel()
    # change model here
    model.phi_min = deg2rad(-45.0)
    model.phi_size = 16
    model.phi_inc = deg2rad(90.0 / 16)

    model.theta_min = -np.pi
    model.theta_size = 50
    model.theta_inc = 2.0 * np.pi / model.theta_size # covering the whole range

    model.range_min = 0.5
    model.range_max = 80.0

    return model

def my_pinhole_model():
    # create a depth image like sensor

    model = PinholeModel()

    model.width = 50
    model.height = 10

    # stretch with 1
    model.f[0] = model.width / 1.25
    model.f[1] = model.height / 1.25

    # set center to optimal center
    model.c[0] = model.width / 2.0
    model.c[1] = model.height / 2.0

    return model

##
# 1. Define sensor model
##
model = my_pinhole_model()
norm_x = False

##
# 2. Generate rays an visualize
##
ray_dirs = gen_ray_dirs(model, norm_x)

# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)

# Creating plot
ax.scatter3D([0.0], [0.0], [0.0], color = "red", label="sensor origin")
ax.scatter3D(ray_dirs[0], ray_dirs[1], ray_dirs[2], color = "green", label="rays")
plt.title("rmagine model visualizatio")
plt.legend()

# show plot
plt.show()


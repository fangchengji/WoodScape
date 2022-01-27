import numpy as np
from scipy.spatial.transform import Rotation as SciRot

#------------------m01------------
# front
alpha = np.pi
beta = (90 + 27) / 180 * np.pi
gama = np.pi / 2

# pitch = 0
# roll = -(90 + 27) / 180 * np.pi
# yaw = -90

x = (27 + 90) / 180 * np.pi
z = np.pi / 2
y = 0

# rear
# alpha = 0
# beta = 57 / 180 * np.pi
# gama = np.pi / 2

# left
# alpha = 0
# beta = 50 / 180 * np.pi
# gama = 0

# right
# alpha = 0
# beta = 55 / 180 * np.pi
# gama = 0

rot_zxz = np.array([alpha, beta, gama])
rotation = SciRot.from_euler(angles=rot_zxz, seq='zxz')

print("rotation: ", rotation.as_matrix())
print("quat: ", rotation.as_quat())

rot_xzy = np.array([x, z, y])
rotation = SciRot.from_euler(angles=rot_xzy, seq='xzy')

print("rotation: ", rotation.as_matrix())
print("quat: ", rotation.as_quat())
import numpy as np
from scipy.spatial.transform import Rotation as SciRot

#------------------m01------------
# front
# alpha = 0
# beta = 65 / 180 * np.pi
# gama = np.pi / 2

# rear
# alpha = 0
# beta = 57 / 180 * np.pi
# gama = np.pi / 2

# left
# alpha = 0
# beta = 50 / 180 * np.pi
# gama = 0

# right
alpha = 0
beta = 55 / 180 * np.pi
gama = 0

rot_zxz = np.array([alpha, beta, gama])
rotation = SciRot.from_euler(angles=rot_zxz, seq='zxz')

print("quat: ", rotation.as_quat())
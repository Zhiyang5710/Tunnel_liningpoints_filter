import laspy
import numpy as np
from tunnel_pcsf import TunnelPCSF

las = laspy.read('data_input\\260028-051初支点云.las')
# las = laspy.read('data_input\\data2-2_RailwayDri_longseg-MoveOutlierR_ROTATED.las')
points = np.vstack([las.x, las.y, las.z]).T  # (N, 3)

csf = TunnelPCSF()
csf.set_point_cloud(points)
csf.preview_initial_cloth(save_path='cloth_preview.png', show=False)
lining_idx, interior_idx = csf.do_filtering()


# 保存结果（带原始属性）
out = laspy.LasData(las.header)
out.points = las.points[lining_idx]
out.write('lining.las')
out.points = las.points[interior_idx]
out.write('interior.las')


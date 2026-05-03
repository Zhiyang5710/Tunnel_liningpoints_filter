# TunnelPCSF

**Cylindrical Cloth Simulation Filter** for tunnel LiDAR point cloud processing.

Separates tunnel **lining points (衬砌点)** from **interior points (非衬砌点)**
using a virtual cylindrical cloth that shrinks radially inward under simulated
radial "gravity" toward the tunnel axis.

---

## Installation

```bash
pip install .
# or in development mode:
pip install -e .
```

---

## Quick Start

```python
import numpy as np
from tunnel_pcsf import TunnelPCSF, TunnelPCSFParams

# Load your tunnel point cloud (N x 3)
points = np.loadtxt("tunnel_scan.txt")

pcsf = TunnelPCSF()
pcsf.params.cloth_resolution_angle = 1.0   # degrees
pcsf.params.cloth_resolution_z = 0.1       # meters
pcsf.params.class_threshold = 0.2          # meters (radial)
pcsf.params.smooth_slope = True

pcsf.set_point_cloud(points)
lining_idx, interior_idx = pcsf.do_filtering()

lining_points   = points[lining_idx]    # 衬砌点
interior_points = points[interior_idx]  # 非衬砌点 (设备、人、螺栓等)

pcsf.save_points(lining_idx,   "lining.txt")
pcsf.save_points(interior_idx, "interior.txt")
```

### With a known tunnel axis

```python
pcsf.set_axis(
    origin    = np.array([0.0, 0.0, 0.0]),
    direction = np.array([0.0, 0.0, 1.0]),  # tunnel runs along Z
)
```

### With laspy

```python
import laspy
import numpy as np
from tunnel_pcsf import TunnelPCSF

las = laspy.read("tunnel.las")
points = np.vstack([las.x, las.y, las.z]).T

pcsf = TunnelPCSF()
pcsf.set_point_cloud(points)
lining_idx, interior_idx = pcsf.do_filtering()
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `cloth_resolution_angle` | `1.0` | Angular resolution of cloth grid (degrees) |
| `cloth_resolution_z` | `0.1` | Axial resolution of cloth grid (meters) |
| `class_threshold` | `0.3` | Radial distance threshold for lining classification (meters) |
| `rigidness` | `2` | Cloth stiffness: 1=soft, 2=medium, 3=rigid |
| `time_step` | `0.65` | Simulation time step |
| `iterations` | `500` | Max simulation iterations |
| `smooth_slope` | `True` | Fill gaps in cloth after simulation |
| `axis_method` | `'pca'` | Axis estimation: `'pca'` or `'provided'` |
| `initial_radius_offset` | `0.5` | How far outside the tunnel the cloth starts (m) | (not necessary, Use scaling factor in some cases , default is r*1.5)

---

## Algorithm Overview

```
1. Estimate tunnel axis (PCA or user-provided)
2. Convert point cloud to cylindrical coordinates (r, θ, z)
3. Initialize a cylindrical cloth mesh at radius = median_r + offset
4. Iteratively:
   a. Apply radial inward "gravity" to each cloth particle (such as Verlet integration)
   b. Detect collisions: pin particles that reach a point cloud surface
5. Post-process: smooth gaps in cloth
6. Classify: points within `class_threshold` of cloth → lining; others → interior
```

---

## Citation

Developed by <Zhiyang ZHI> at the Sun Yat-sen University, China Zhuhai.
P-CSF is an algorithm for multi-type tunnel point clouds. Adapted from [CSF](https://github.com/jianboqi/CSF) (Zhang et al., 2016).

If you use this work, please also cite the original P-CSF paper:

> Zhi Z, Chang B, Li Y,et al.P-CSF: Polar coordinate cloth simulation filtering algorithm for multi-type tunnel point clouds[J].*Tunnelling and underground space technology*, 2025(Jan. Pt.1):155.DOI:10.1016/j.tust.2024.106144.




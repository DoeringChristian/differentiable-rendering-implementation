"""
Optimization without regularization.
Leads to self intersections.
"""


import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from mitsuba.scalar_rgb import Transform4f as T
import numpy as np
import trimesh
import tqdm

mi.set_variant("cuda_ad_rgb")

import util

suzanne = mi.load_dict(
    {
        "type": "ply",
        "filename": "data/suzanne.ply",
        "to_world": T.scale(0.5),
        "face_normals": True,
    }
)

scene = {
    "type": "scene",
    "integrator": {
        "type": "path",
    },
    "sensor": {
        "type": "perspective",
        "to_world": T.look_at(origin=(0, 2, 2), target=(0, 0, 0), up=(0, 0, 1)),
        "film": {
            "type": "hdrfilm",
            "width": 1024,
            "height": 1024,
        },
    },
    # "mesh": util.trimesh2mitsuba(trimesh.creation.icosphere()),
    # "mesh": suzanne,
    "floor": {
        "type": "rectangle",
        "to_world": T.translate([0, 0, -1]).scale(10),
    },
    "light": {
        "type": "point",
        "position": [1, 2, 2],
        "intensity": {
            "type": "spectrum",
            "value": 10.0,
        },
    },
}

scene_ref = mi.load_dict({**scene, **{"mesh": suzanne}})

ref = mi.render(scene_ref, spp=128)

scene = mi.load_dict(
    {
        **scene,
        **{
            "mesh": util.trimesh2mitsuba(
                trimesh.creation.icosphere(subdivisions=4, radius=0.5)
            )
        },
    }
)

params = mi.traverse(scene)
print(params)
params.keep("mesh.vertex_positions")

opt = mi.ad.Adam(lr=0.003, params=params)
print(f"{opt=}")

params.update(opt)

for i in tqdm.tqdm(range(50)):
    img = mi.render(scene, params, spp=1)

    mi.util.write_bitmap(f"out/{i}.jpg", img)

    loss = dr.mean(dr.sqr(img - ref))

    dr.backward(loss)

    opt.step()

    params.update(opt)

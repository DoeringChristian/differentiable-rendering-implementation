import mitsuba as mi
import matplotlib.pyplot as plt
import drjit as dr

mi.set_variant("cuda_ad_rgb")

scene: mi.Scene = mi.load_file("data/scene.xml")
integrator = mi.load_dict({"type": "prb_reparam"})

params = mi.traverse(scene)
print(f"{params=}")

key = "PLYMesh.vertex_positions"

initial_pos = dr.unravel(mi.Point3f, params[key])

ref = mi.render(scene, spp=1024)

mi.util.write_bitmap("out/ref.jpg", ref)

opt = mi.ad.Adam(lr=0.025)
opt["pos"] = mi.Vector3f(1, 0, 0)
opt["axis"] = mi.Vector3f(0, 1, 0)
opt["angle"] = mi.Float(90)


def apply_transform(params, opt):
    pos = opt["pos"]
    axis = opt["axis"]
    angle = opt["angle"]

    params[key] = dr.ravel(
        mi.Transform4f.translate(pos).rotate(axis, angle) @ initial_pos
    )
    params.update()


losses = []

apply_transform(params, opt)

for i in range(50):
    print(f"Iteration {i}:")
    print("Rendering Image...")
    img = mi.render(scene, params, integrator=integrator, seed=i, spp=16)

    loss = dr.mean(dr.sqr(img - ref))

    print("Backpropagating Gradients...")
    dr.backward(loss)

    print("Updating Optimizer...")
    opt.step()

    print("Updating Parameters...")
    apply_transform(params, opt)

    print("Writing output image...")
    mi.util.write_bitmap(f"out/{i}.jpg", img)

    losses.append(loss)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0][0].plot(losses)

axs[0][1].imshow(mi.util.convert_to_bitmap(img))

axs[1][1].imshow(mi.util.convert_to_bitmap(ref))

plt.show()

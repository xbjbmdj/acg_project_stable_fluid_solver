Blender batch rendering helper

Files added:
- scripts/render_blender.py — Blender Python script to import an OBJ, set alpha, camera, samples, and render to PNG.
- scripts/batch_render.sh — Shell wrapper to call Blender headlessly over a directory of .obj files.

Basic usage examples:

1) Single render with custom params (from repo root):

```bash
blender --background --python scripts/render_blender.py -- --obj build/model.obj --out_dir renders --alpha 0.2 --cam 200,-200,200 --cam_rot 54.736,-0.000003,45 --look_at 0,0,0 --samples 512 --res 1920,1080
```

2) Batch render all OBJs in a folder:

```bash
./scripts/batch_render.sh build renders --alpha 0.2 --cam 200,-200,200 --cam_rot 54.736,-0.000003,45 --look_at 0,0,0 --samples 512 --res 1920,1080
```

Notes:
- Requires Blender CLI available as `blender`.
- `render_blender.py` sets `film_transparent` and outputs PNG with RGBA.
- If Blender is not installed on this machine, run the scripts on a machine that has Blender.

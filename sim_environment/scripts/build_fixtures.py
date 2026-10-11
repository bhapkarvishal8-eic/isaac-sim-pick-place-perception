import omni.usd
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf

print("FIXTURES_MARKER_0_STARTED", flush=True)

FIXTURES_DIR = "/home/vishal/poc/stage1_perception_arm/phase2_sim_environment/fixtures"

def new_stage_at(path):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage

def make_material(stage, path, color, roughness=0.8, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat

# ---------------------------------------------------------------------------
# PILL prototype: cylinder, 8mm diameter (radius 4mm) x 3mm height.
# Multiple colors are achieved by overriding displayColor per-instance when
# this prototype is referenced/instanced in the workspace stage, not baked
# into the prototype itself (single reusable prototype, per the kickoff doc's
# explicit requirement).
# ---------------------------------------------------------------------------
stage = new_stage_at(f"{FIXTURES_DIR}/pill.usd")
pill_root = stage.DefinePrim("/Pill", "Xform")
stage.SetDefaultPrim(pill_root)
cyl = UsdGeom.Cylinder.Define(stage, "/Pill/geom")
cyl.CreateRadiusAttr(0.004)   # 8mm diameter -> 4mm radius
cyl.CreateHeightAttr(0.003)   # 3mm height
cyl.CreateAxisAttr("Z")
cyl.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.85, 0.9)])  # default/neutral, overridden per-instance
UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(pill_root)
UsdPhysics.MassAPI.Apply(pill_root).CreateMassAttr(0.0005)  # ~0.5g, plausible for a small pill
stage.GetRootLayer().Save()
print(f"FIXTURES_MARKER_1_PILL_SAVED path={FIXTURES_DIR}/pill.usd", flush=True)

# ---------------------------------------------------------------------------
# BOX prototype: cuboid, 3 discrete sizes (10/15/20cm sides) via a Size
# variantSet, cardboard-style material.
# ---------------------------------------------------------------------------
stage = new_stage_at(f"{FIXTURES_DIR}/box.usd")
box_root = stage.DefinePrim("/Box", "Xform")
stage.SetDefaultPrim(box_root)
cube = UsdGeom.Cube.Define(stage, "/Box/geom")
cube.CreateSizeAttr(1.0)  # unit cube; scaled per Size variant below

cardboard_mat = make_material(stage, "/Box/CardboardMaterial", color=(0.72, 0.55, 0.35), roughness=0.95, metallic=0.0)
UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(cardboard_mat)

UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(box_root)

vset = box_root.GetVariantSets().AddVariantSet("Size")
for name, side_m in [("Small_10cm", 0.10), ("Medium_15cm", 0.15), ("Large_20cm", 0.20)]:
    vset.AddVariant(name)
    vset.SetVariantSelection(name)
    with vset.GetVariantEditContext():
        cube.AddScaleOp().Set(Gf.Vec3f(side_m, side_m, side_m))
        UsdPhysics.MassAPI.Apply(box_root).CreateMassAttr(0.05 * (side_m / 0.10) ** 3)
vset.SetVariantSelection("Medium_15cm")
stage.GetRootLayer().Save()
print(f"FIXTURES_MARKER_2_BOX_SAVED path={FIXTURES_DIR}/box.usd variants={vset.GetVariantNames()}", flush=True)

# ---------------------------------------------------------------------------
# SHELF prototype: static collision mesh with compartments (3x2 cubby grid).
# ---------------------------------------------------------------------------
stage = new_stage_at(f"{FIXTURES_DIR}/shelf.usd")
shelf_root = stage.DefinePrim("/Shelf", "Xform")
stage.SetDefaultPrim(shelf_root)

shelf_mat = make_material(stage, "/Shelf/ShelfMaterial", color=(0.55, 0.55, 0.58), roughness=0.6, metallic=0.1)

# Overall footprint: 0.48m wide x 0.25m deep x 0.32m tall, 3 columns x 2 rows of
# compartments. Sized to fit inside the fixed 0.8m-high camera's FOV footprint
# at shelf-top height (see build_scene.py's FOV coverage check) -- the camera
# height is a hard spec requirement, so the shelf is sized to the achievable
# coverage rather than the other way around.
outer_w, outer_d, outer_h = 0.48, 0.25, 0.32
wall_t = 0.01
cols, rows = 3, 2
compartment_w = (outer_w - wall_t * (cols + 1)) / cols
compartment_h = (outer_h - wall_t * (rows + 1)) / rows

def add_panel(name, size_xyz, center_xyz):
    prim = UsdGeom.Cube.Define(stage, f"/Shelf/{name}")
    prim.CreateSizeAttr(1.0)
    prim.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    prim.AddTranslateOp().Set(Gf.Vec3f(*center_xyz))
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(shelf_mat)
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    return prim

# Back panel
add_panel("back_panel", (outer_w, wall_t, outer_h), (0, outer_d / 2 - wall_t / 2, outer_h / 2))
# Base panel
add_panel("base_panel", (outer_w, outer_d, wall_t), (0, 0, wall_t / 2))
# Top panel
add_panel("top_panel", (outer_w, outer_d, wall_t), (0, 0, outer_h - wall_t / 2))
# Vertical dividers (cols+1 of them, forming `cols` compartments across)
for c in range(cols + 1):
    x = -outer_w / 2 + wall_t / 2 + c * (compartment_w + wall_t)
    add_panel(f"divider_v_{c}", (wall_t, outer_d, outer_h), (x, 0, outer_h / 2))
# Horizontal shelves (rows+1 of them, forming `rows` compartments vertically)
for r in range(rows + 1):
    z = wall_t / 2 + r * (compartment_h + wall_t)
    add_panel(f"divider_h_{r}", (outer_w, outer_d, wall_t), (0, 0, z))

# Static: mark the whole shelf as a fixed/static collider (no RigidBodyAPI -> static collider by default in USD Physics)
stage.GetRootLayer().Save()
print(f"FIXTURES_MARKER_3_SHELF_SAVED path={FIXTURES_DIR}/shelf.usd compartments={cols}x{rows}", flush=True)

print("FIXTURES_MARKER_4_DONE", flush=True)

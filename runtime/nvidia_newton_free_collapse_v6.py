import hashlib
import json
import math
import pathlib
import platform

import newton
import warp as wp

wp.init()

R = 0.035
NX, NY, NZ = 4, 4, 8
SPACING = 2.08 * R
FPS = 120
SUBSTEPS = 4
FRAMES = 480
DT = 1.0 / FPS / SUBSTEPS
SNAPSHOT_FRAMES = {60, 120, 240, 480}


def metrics(state):
    q = state.body_q.numpy()
    xs = [float(v[0]) for v in q]
    ys = [float(v[1]) for v in q]
    zs = [float(v[2]) for v in q]
    return {
        "spread_x_m": max(xs) - min(xs),
        "spread_y_m": max(ys) - min(ys),
        "spread_m": max(max(xs) - min(xs), max(ys) - min(ys)),
        "height_m": max(zs) - min(zs),
        "com_z_m": sum(zs) / len(zs),
        "z_min_m": min(zs),
        "z_max_m": max(zs),
        "grains_below_3r": sum(z < 3.0 * R for z in zs),
    }


def run_case(g):
    cfg = newton.ModelBuilder.ShapeConfig(
        mu=0.70,
        restitution=0.02,
        ke=3e4,
        kd=120.0,
        density=1800.0,
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -g))
    builder.add_ground_plane(cfg=cfg)

    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                jitter_x = (((i * 17 + j * 13 + k * 7) % 11) - 5) * 0.0007
                jitter_y = (((i * 11 + j * 19 + k * 5) % 13) - 6) * 0.0007
                x = (i - (NX - 1) / 2.0) * SPACING + jitter_x
                y = (j - (NY - 1) / 2.0) * SPACING + jitter_y
                z = R + 0.004 + k * SPACING
                body = builder.add_body(xform=wp.transform((x, y, z), wp.quat_identity()))
                builder.add_shape_sphere(body, radius=R, cfg=cfg)

    model = builder.finalize()
    state0 = model.state()
    state1 = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model, iterations=16, enable_restitution=True)
    snapshots = {}
    max_contacts = 0

    for frame in range(1, FRAMES + 1):
        for _ in range(SUBSTEPS):
            state0.clear_forces()
            pipeline.collide(state0, contacts)
            max_contacts = max(max_contacts, int(contacts.rigid_contact_count.numpy()[0]))
            solver.step(state0, state1, control, contacts, DT)
            state0, state1 = state1, state0
        if frame in SNAPSHOT_FRAMES:
            snapshots[f"{frame / FPS:.1f}s"] = metrics(state0)

    return {
        "gravity_m_s2": g,
        "max_rigid_contacts": max_contacts,
        "snapshots": snapshots,
        "final": metrics(state0),
    }


def relative_delta(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


lunar = run_case(1.62)
earth = run_case(9.80665)

comparisons = {}
deltas = []
for key in ("0.5s", "1.0s", "2.0s", "4.0s"):
    lm = lunar["snapshots"][key]
    em = earth["snapshots"][key]
    per = {}
    for metric in ("spread_m", "height_m", "com_z_m"):
        d = relative_delta(lm[metric], em[metric])
        per[metric + "_relative_delta"] = d
        deltas.append(d)
    comparisons[key] = per

discrimination_index = sum(deltas) / len(deltas)
finite_values = [
    discrimination_index,
    lunar["final"]["spread_m"],
    earth["final"]["spread_m"],
    lunar["final"]["com_z_m"],
    earth["final"]["com_z_m"],
]
executed_ok = (
    lunar["max_rigid_contacts"] > 0
    and earth["max_rigid_contacts"] > 0
    and all(math.isfinite(x) for x in finite_values)
)

out = {
    "farm": 139,
    "engine": "NVIDIA Newton",
    "version": getattr(newton, "__version__", "unknown"),
    "solver": "SolverXPBD",
    "test": "FREE_GRANULAR_COLUMN_COLLAPSE_V6",
    "grain_count": NX * NY * NZ,
    "radius_m": R,
    "density_kg_m3": 1800.0,
    "friction_mu": 0.70,
    "frames": FRAMES,
    "substeps": SUBSTEPS,
    "steps_per_case": FRAMES * SUBSTEPS,
    "snapshot_times_s": [0.5, 1.0, 2.0, 4.0],
    "lunar": lunar,
    "earth": earth,
    "comparisons": comparisons,
    "dynamic_discrimination_index": discrimination_index,
    "discriminating_over_10pct_proxy": discrimination_index > 0.10,
    "status": "EXECUTED" if executed_ok else "FAIL",
    "scope": "REAL_NEWTON_FREE_COLUMN_COLLAPSE_TRANSIENT_ABLATION_NOT_CALIBRATED_REGOLITH_DEM_OR_PHYSICAL_VALIDATION",
    "limitations": [
        "monodisperse spheres",
        "numerical material/contact parameters are not experimentally calibrated",
        "discrimination index is an engineering sensitivity proxy, not a physical validation metric",
        "fixed real-time comparison intentionally tests transient gravity sensitivity",
    ],
    "python": platform.python_version(),
}

raw = json.dumps(out, sort_keys=True).encode()
out["result_sha256"] = hashlib.sha256(raw).hexdigest()
pathlib.Path("artifacts").mkdir(exist_ok=True)
pathlib.Path("artifacts/f139_newton_free_collapse_v6.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
raise SystemExit(0 if executed_ok else 1)

import hashlib
import json
import math
import pathlib
import platform

import newton
import warp as wp

wp.init()

RADIUS = 0.035
GRID = (4, 4, 8)
SPACING = 2.08 * RADIUS
FPS = 120
SUB = 4
FRAMES = 480
DT = 1.0 / FPS / SUB
CHECK_FRAMES = (60, 120, 240, 480)

REFERENCE_V6 = {
    "run_id": 35899534732,
    "artifact_id": 10767758851,
    "artifact_digest": "sha256:6dcf42b7a4302975a90223c39f3e55fe03ee5c38060c538c7776e59ebdc0c9b1",
    "result_sha256": "e5beddcd3609538d4dc9ef882f3c26af6bc62427b83a4773eb05b199f9a9ee4a",
    "lunar_final_spread_m": 2.8939764499664307,
    "earth_final_spread_m": 6.453675746917725,
    "discrimination_index": 0.387336969872355,
}


def summarize(state):
    q = state.body_q.numpy()
    xyz = [(float(v[0]), float(v[1]), float(v[2])) for v in q]
    xs = [p[0] for p in xyz]
    ys = [p[1] for p in xyz]
    zs = [p[2] for p in xyz]
    return {
        "spread_m": max(max(xs) - min(xs), max(ys) - min(ys)),
        "height_m": max(zs) - min(zs),
        "com_z_m": sum(zs) / len(zs),
        "z_min_m": min(zs),
        "z_max_m": max(zs),
        "grains_below_3r": sum(z < 3.0 * RADIUS for z in zs),
    }


def initial_position(i, j, k):
    jx = (((i * 17 + j * 13 + k * 7) % 11) - 5) * 0.0007
    jy = (((i * 11 + j * 19 + k * 5) % 13) - 6) * 0.0007
    return (
        (i - (GRID[0] - 1) / 2.0) * SPACING + jx,
        (j - (GRID[1] - 1) / 2.0) * SPACING + jy,
        RADIUS + 0.004 + k * SPACING,
    )


def simulate(gravity):
    cfg = newton.ModelBuilder.ShapeConfig(
        mu=0.70, restitution=0.02, ke=3e4, kd=120.0, density=1800.0
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
    builder.add_ground_plane(cfg=cfg)
    for k in range(GRID[2]):
        for j in range(GRID[1]):
            for i in range(GRID[0]):
                body = builder.add_body(
                    xform=wp.transform(initial_position(i, j, k), wp.quat_identity())
                )
                builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)

    model = builder.finalize()
    a, b = model.state(), model.state()
    control = model.control()
    collision = newton.CollisionPipeline(model)
    contacts = collision.contacts()
    solver = newton.solvers.SolverXPBD(model, iterations=16, enable_restitution=True)

    max_contacts = 0
    snapshots = {}
    for frame in range(1, FRAMES + 1):
        for _ in range(SUB):
            a.clear_forces()
            collision.collide(a, contacts)
            max_contacts = max(max_contacts, int(contacts.rigid_contact_count.numpy()[0]))
            solver.step(a, b, control, contacts, DT)
            a, b = b, a
        if frame in CHECK_FRAMES:
            snapshots[f"{frame / FPS:.1f}s"] = summarize(a)

    return {
        "gravity_m_s2": gravity,
        "max_rigid_contacts": max_contacts,
        "snapshots": snapshots,
        "final": summarize(a),
    }


def rel_delta(x, y):
    return abs(x - y) / max(abs(x), abs(y), 1e-12)


lunar = simulate(1.62)
earth = simulate(9.80665)
components = []
for stamp in ("0.5s", "1.0s", "2.0s", "4.0s"):
    for metric in ("spread_m", "height_m", "com_z_m"):
        components.append(rel_delta(lunar["snapshots"][stamp][metric], earth["snapshots"][stamp][metric]))
index = sum(components) / len(components)

checks = {
    "lunar_spread_relative_error": rel_delta(lunar["final"]["spread_m"], REFERENCE_V6["lunar_final_spread_m"]),
    "earth_spread_relative_error": rel_delta(earth["final"]["spread_m"], REFERENCE_V6["earth_final_spread_m"]),
    "discrimination_index_relative_error": rel_delta(index, REFERENCE_V6["discrimination_index"]),
}
reproduced = (
    lunar["max_rigid_contacts"] > 0
    and earth["max_rigid_contacts"] > 0
    and all(math.isfinite(v) for v in checks.values())
    and max(checks.values()) < 1e-6
)

out = {
    "farm": 139,
    "engine": "NVIDIA Newton",
    "version": getattr(newton, "__version__", "unknown"),
    "test": "FREE_GRANULAR_COLUMN_COLLAPSE_V6_REPRODUCTION_B",
    "reference": REFERENCE_V6,
    "lunar": lunar,
    "earth": earth,
    "dynamic_discrimination_index": index,
    "comparison_to_reference": checks,
    "status": "REPRODUCED" if reproduced else "REPRODUCTION_MISMATCH",
    "maturity": "S7_REPRODUCED" if reproduced else "S4_EXECUTED",
    "limitations": [
        "same Newton engine and same numerical parameters as V6, separate workflow and implementation",
        "reproduction reduces run-level error but is not an independent physical reference",
        "monodisperse uncalibrated spheres; not calibrated regolith DEM",
    ],
    "python": platform.python_version(),
}
raw = json.dumps(out, sort_keys=True).encode()
out["result_sha256"] = hashlib.sha256(raw).hexdigest()
pathlib.Path("artifacts").mkdir(exist_ok=True)
pathlib.Path("artifacts/f139_newton_free_collapse_v6_reproduction_b.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
print(json.dumps(out))
raise SystemExit(0 if reproduced else 1)

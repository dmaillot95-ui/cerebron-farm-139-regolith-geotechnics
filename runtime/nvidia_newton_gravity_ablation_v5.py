import json,hashlib,pathlib,platform,math
import warp as wp,newton
wp.init();r=.035;nx=5;ny=5;nz=4;spacing=.078;fps=120;sub=4;frames=300;dt=1/fps/sub

def run(g):
 cfg=newton.ModelBuilder.ShapeConfig(mu=.70,restitution=.02,ke=3e4,kd=120.,density=1800.)
 b=newton.ModelBuilder(gravity=(0,0,-g));b.add_ground_plane(cfg=cfg)
 for k in range(nz):
  for j in range(ny):
   for i in range(nx):
    body=b.add_body(xform=wp.transform(((i-2)*spacing,(j-2)*spacing,.25+k*spacing),wp.quat_identity()));b.add_shape_sphere(body,radius=r,cfg=cfg)
 m=b.finalize();s0=m.state();s1=m.state();ctrl=m.control();pipe=newton.CollisionPipeline(m);contacts=pipe.contacts();solver=newton.solvers.SolverXPBD(m,iterations=16,enable_restitution=True);mc=0
 for _ in range(frames):
  for _ in range(sub):
   s0.clear_forces();pipe.collide(s0,contacts);mc=max(mc,int(contacts.rigid_contact_count.numpy()[0]));solver.step(s0,s1,ctrl,contacts,dt);s0,s1=s1,s0
 q=s0.body_q.numpy();xs=[float(a[0]) for a in q];ys=[float(a[1]) for a in q];zs=[float(a[2]) for a in q]
 return {"g":g,"zmin":min(zs),"zmax":max(zs),"spread":max(max(xs)-min(xs),max(ys)-min(ys)),"contacts":mc,"low":sum(z<3*r for z in zs)}
lunar=run(1.62);earth=run(9.80665);ratio=earth["zmax"]/lunar["zmax"];ok=lunar["contacts"]>0 and earth["contacts"]>0 and all(math.isfinite(x) for x in [ratio,lunar["spread"],earth["spread"]])
out={"farm":139,"engine":"NVIDIA Newton","version":getattr(newton,"__version__","unknown"),"test":"GRAVITY_ABLATION_LUNAR_VS_EARTH","grain_count":100,"steps_per_case":frames*sub,"lunar":lunar,"earth":earth,"earth_to_lunar_zmax_ratio":ratio,"status":"ABLATION_OK" if ok else "FAIL","scope":"CONTROLLED_GRAVITY_ABLATION_SAME_NUMERICAL_MODEL_NOT_EXPERIMENTAL_CALIBRATION","python":platform.python_version()};raw=json.dumps(out,sort_keys=True).encode();out["result_sha256"]=hashlib.sha256(raw).hexdigest();pathlib.Path("artifacts").mkdir(exist_ok=True);pathlib.Path("artifacts/f139_newton_gravity_ablation_v5.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out));raise SystemExit(0 if ok else 1)

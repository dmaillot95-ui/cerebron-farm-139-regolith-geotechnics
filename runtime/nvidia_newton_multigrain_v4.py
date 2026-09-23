import json,hashlib,pathlib,platform,math
import warp as wp,newton
wp.init();g=1.62;r=.035;nx=5;ny=5;nz=4;spacing=.078;fps=120;sub=4;frames=300;dt=1/fps/sub
cfg=newton.ModelBuilder.ShapeConfig(mu=0.70,restitution=0.02,ke=3e4,kd=120.0,density=1800.0)
b=newton.ModelBuilder(gravity=(0,0,-g));b.add_ground_plane(cfg=cfg)
for k in range(nz):
 for j in range(ny):
  for i in range(nx):
   x=(i-(nx-1)/2)*spacing;y=(j-(ny-1)/2)*spacing;z=.25+k*spacing
   body=b.add_body(xform=wp.transform((x,y,z),wp.quat_identity()));b.add_shape_sphere(body,radius=r,cfg=cfg)
m=b.finalize();s0=m.state();s1=m.state();ctrl=m.control();pipe=newton.CollisionPipeline(m);contacts=pipe.contacts();solver=newton.solvers.SolverXPBD(m,iterations=16,enable_restitution=True);maxc=0
for _ in range(frames):
 for _ in range(sub):
  s0.clear_forces();pipe.collide(s0,contacts);maxc=max(maxc,int(contacts.rigid_contact_count.numpy()[0]));solver.step(s0,s1,ctrl,contacts,dt);s0,s1=s1,s0
q=s0.body_q.numpy();xs=[float(a[0]) for a in q];ys=[float(a[1]) for a in q];zs=[float(a[2]) for a in q];zmin=min(zs);zmax=max(zs);spread=max(max(xs)-min(xs),max(ys)-min(ys));settled=sum(1 for z in zs if z<3*r);ok=maxc>=len(zs) and zmin>0.8*r and zmin<1.3*r and settled>0
out={"farm":139,"engine":"NVIDIA Newton","version":getattr(newton,"__version__","unknown"),"solver":"SolverXPBD","test":"LUNAR_100_GRAIN_SETTLING","grain_count":len(zs),"radius_m":r,"density_kg_m3":1800.0,"friction_mu":.70,"gravity_m_s2":g,"steps":frames*sub,"max_rigid_contacts":maxc,"final_z_min_m":zmin,"final_z_max_m":zmax,"horizontal_spread_m":spread,"grains_below_3r":settled,"status":"MULTIGRAIN_CONTACT_OK" if ok else "FAIL","scope":"REAL_NEWTON_MULTI_RIGID_GRAIN_CONTACT_SIMULATION_NOT_CALIBRATED_REGOLITH_DEM_OR_PHYSICAL_VALIDATION","python":platform.python_version()};raw=json.dumps(out,sort_keys=True).encode();out["result_sha256"]=hashlib.sha256(raw).hexdigest();pathlib.Path("artifacts").mkdir(exist_ok=True);pathlib.Path("artifacts/f139_newton_multigrain_v4.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out));raise SystemExit(0 if ok else 1)

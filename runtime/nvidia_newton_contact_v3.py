import json,hashlib,pathlib,platform,math
import warp as wp,newton
wp.init();g=1.62;radius=.08;z0=1.0;fps=120;sub=4;frames=180;dt=1/fps/sub
cfg=newton.ModelBuilder.ShapeConfig(mu=0.65,restitution=0.05,ke=2e4,kd=100.0,density=1800.0)
b=newton.ModelBuilder(gravity=(0.0,0.0,-g));b.add_ground_plane(cfg=cfg);body=b.add_body(xform=wp.transform((0,0,z0),wp.quat_identity()));b.add_shape_sphere(body,radius=radius,cfg=cfg);m=b.finalize();s0=m.state();s1=m.state();ctrl=m.control();pipe=newton.CollisionPipeline(m);contacts=pipe.contacts();solver=newton.solvers.SolverXPBD(m,iterations=12,enable_restitution=True);maxc=0
for _ in range(frames):
 for _ in range(sub):
  s0.clear_forces();pipe.collide(s0,contacts);maxc=max(maxc,int(contacts.rigid_contact_count.numpy()[0]));solver.step(s0,s1,ctrl,contacts,dt);s0,s1=s1,s0
q=s0.body_q.numpy();zf=float(q[0][2]);expected=radius;err=abs(zf-expected);ok=maxc>0 and err<0.03
out={"farm":139,"engine":"NVIDIA Newton","version":getattr(newton,"__version__","unknown"),"solver":"SolverXPBD","collision":"Newton CollisionPipeline","test":"LUNAR_SPHERE_GROUND_CONTACT","gravity_m_s2":g,"radius_m":radius,"initial_height_m":z0,"frames":frames,"substeps":sub,"steps":frames*sub,"max_rigid_contacts":maxc,"final_center_z_m":zf,"expected_rest_center_z_m":expected,"rest_height_abs_error_m":err,"status":"NATIVE_CONTACT_REFERENCE_OK" if ok else "FAIL","scope":"REAL_NEWTON_RIGID_CONTACT_DYNAMICS_REFERENCE_NOT_REGOLITH_GRANULAR_CALIBRATION_OR_PHYSICAL_VALIDATION","python":platform.python_version()};raw=json.dumps(out,sort_keys=True).encode();out["result_sha256"]=hashlib.sha256(raw).hexdigest();pathlib.Path("artifacts").mkdir(exist_ok=True);pathlib.Path("artifacts/f139_newton_contact_v3.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out));raise SystemExit(0 if ok else 1)

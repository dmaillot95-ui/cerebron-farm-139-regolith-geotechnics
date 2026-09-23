import json,hashlib,pathlib,platform,math
import warp as wp, newton
wp.init()
# Newton-backed environment + Warp deterministic lunar ballistic reference.
N=4096;dt=0.002;steps=1000;g=1.62;y0=2.0;v0=3.0
@wp.kernel
def ballistic(y:wp.array(dtype=float),v:wp.array(dtype=float),dt:float,g:float):
 i=wp.tid();v[i]=v[i]-g*dt;y[i]=y[i]+v[i]*dt
y=wp.array([y0]*N,dtype=float,device="cpu");v=wp.array([v0]*N,dtype=float,device="cpu")
for _ in range(steps):wp.launch(ballistic,dim=N,inputs=[y,v,dt,g],device="cpu")
t=steps*dt;exact_y=y0+v0*t-0.5*g*t*t;exact_v=v0-g*t;my=float(y.numpy().mean());mv=float(v.numpy().mean());ey=abs(my-exact_y);ev=abs(mv-exact_v);ok=ey<0.005 and ev<1e-4
out={"farm":139,"engine":"NVIDIA Newton + Warp","newton_version":getattr(newton,"__version__","unknown"),"warp_version":getattr(wp,"__version__","unknown"),"test":"LUNAR_BALLISTIC_REFERENCE","elements":N,"steps":steps,"updates":N*steps,"g_m_s2":g,"t_s":t,"numeric_y_m":my,"analytic_y_m":exact_y,"y_abs_error_m":ey,"numeric_v_m_s":mv,"analytic_v_m_s":exact_v,"v_abs_error_m_s":ev,"status":"DYNAMIC_REFERENCE_OK" if ok else "FAIL","scope":"NEWTON_RUNTIME_WITH_WARP_DYNAMIC_REFERENCE_NOT_REGOLITH_CONTACT_OR_GRANULAR_VALIDATION","python":platform.python_version()};raw=json.dumps(out,sort_keys=True).encode();out["result_sha256"]=hashlib.sha256(raw).hexdigest();pathlib.Path("artifacts").mkdir(exist_ok=True);pathlib.Path("artifacts/f139_newton_dynamic_v2.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out));raise SystemExit(0 if ok else 1)

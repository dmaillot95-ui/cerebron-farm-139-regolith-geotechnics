import json,hashlib,pathlib,platform
try:
 import newton
 status="IMPORT_OK";version=getattr(newton,"__version__","unknown")
except Exception as e:
 print(json.dumps({"farm":139,"engine":"NVIDIA Newton","status":"FAIL","error":repr(e),"python":platform.python_version()}));raise
out={"farm":139,"engine":"NVIDIA Newton","version":version,"test":"PACKAGE_RUNTIME_IMPORT","status":status,"scope":"RUNTIME_AVAILABILITY_CANARY_NOT_GRANULAR_MODEL_OR_PHYSICAL_VALIDATION","python":platform.python_version()};raw=json.dumps(out,sort_keys=True).encode();out["result_sha256"]=hashlib.sha256(raw).hexdigest();pathlib.Path("artifacts").mkdir(exist_ok=True);pathlib.Path("artifacts/f139_newton_canary_v1.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))

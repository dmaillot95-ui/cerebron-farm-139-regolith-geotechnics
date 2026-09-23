import json,inspect,newton
names=[n for n in dir(newton) if any(k in n.lower() for k in ["model","solver","contact","particle","builder","body","shape"])]
out={"version":getattr(newton,"__version__","unknown"),"symbols":names[:250]}
print(json.dumps(out))

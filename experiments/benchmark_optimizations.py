from src import runtime
runtime.configure()
import torch,time,numpy as np,json
from src.model import SmallFCN
from src.utils import load_checkpoint
m=load_checkpoint(SmallFCN(),'params/model_fp16.pt',torch.device('cpu'))
x=torch.rand(1,3,256,256)
with torch.inference_mode():
 candidates={'eager':m,'frozen':torch.jit.freeze(torch.jit.trace(m,x))}
 candidates['optimized']=torch.jit.optimize_for_inference(candidates['frozen'])
 results={}
 ref=m(x)
 for name,model in candidates.items():
  model(x);t=time.process_time()
  for _ in range(100): y=model(x)
  results[name]=dict(ms=(time.process_time()-t)*10,max_logit_error=float((y-ref).abs().max()),different_pixels=int((y.argmax(1)!=ref.argmax(1)).sum()))
 print(json.dumps(results,indent=2))

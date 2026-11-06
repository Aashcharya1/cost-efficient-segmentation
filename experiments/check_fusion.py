from src import runtime
runtime.configure()
import torch,time,cv2,json
from src.utils import load_checkpoint
from src.model import SmallFCN
from src.dataset import DUTSDataset
with torch.inference_mode():
 image=cv2.resize(cv2.imread(str(DUTSDataset('data','val').pairs[0][0])),(256,256),interpolation=cv2.INTER_AREA)
 x=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float()/255
 # Runtime comparison on a validation photo only.
 m=load_checkpoint(SmallFCN(),'experiments/full_optimized.pt',torch.device('cpu'))
 records=[]
 for fusion in [False,True]:
  torch.jit.enable_onednn_fusion(fusion)
  for mode in ['frozen','optimized']:
   try:
    script=torch.jit.freeze(torch.jit.trace(m,x));script=torch.jit.optimize_for_inference(script) if mode=='optimized' else script
    script(x);script(x);t=time.process_time()
    for _ in range(50):script(x)
    r=dict(fusion=fusion,mode=mode,ms=(time.process_time()-t)*20);print(r,flush=True);records.append(r)
   except RuntimeError as e:print('Unavailable',fusion,mode,str(e)[:250],flush=True)

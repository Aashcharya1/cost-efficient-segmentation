from src import runtime
runtime.configure()
import torch,time,json,cv2
from src.model import SmallFCN
from src.utils import load_checkpoint
from src.dataset import DUTSDataset
m=load_checkpoint(SmallFCN(),'params/model_fp16.pt',torch.device('cpu'));x=torch.zeros(1,256,256,3).permute(0,3,1,2)
with torch.inference_mode():
 frozen=torch.jit.freeze(torch.jit.trace(m,x));models={'eager':m,'frozen':frozen,'optimized':torch.jit.optimize_for_inference(frozen)}
 photos=[cv2.resize(cv2.imread(str(ip)),(256,256),interpolation=cv2.INTER_AREA) for ip,mp in DUTSDataset('data','val').pairs[:96]]
 results={name:[] for name in models}
 for model in models.values():model(x)
 for photo in photos:
  for name,model in models.items():
   start=time.process_time();rgb=cv2.cvtColor(photo,cv2.COLOR_BGR2RGB);t=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().div_(255);y=(model(t)[0,1]>model(t)[0,0]).numpy().astype('uint8');results[name].append((time.process_time()-start)*1000)
 print({k:sum(v)/len(v) for k,v in results.items()},flush=True)

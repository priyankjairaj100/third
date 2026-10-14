"""Slurm-only A100 capability probe, no model or data loading."""
import hashlib,importlib.metadata,json,os,platform,socket,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
p=Path(__file__).resolve().parent;pin=sys.argv[1];out=Path(sys.argv[2])
raw=(p/'package_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==pin
for name,b in json.loads(raw)['files'].items():
 data=(p/name).read_bytes();assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==b
assert os.environ.get('SLURM_JOB_ID','').isdigit() and socket.gethostname()=='csis.cn4'
pins=json.loads((p/'runtime_expectations.json').read_bytes());assert platform.python_version()==pins['python']
versions={name:importlib.metadata.version(name) for name in pins['package_versions']};assert versions==pins['package_versions']
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
device=torch.cuda.get_device_properties(0)
assert 'A100' in device.name and device.total_memory>=75*1024**3 and torch.cuda.get_device_capability(0)==(8,0)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
a=torch.zeros((1,2,128,64),device='cuda',dtype=torch.bfloat16)
with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
 result=torch.nn.functional.scaled_dot_product_attention(a,a,a,dropout_p=0.0)
 flags={'flash':torch.backends.cuda.flash_sdp_enabled(),'math':torch.backends.cuda.math_sdp_enabled(),'mem_efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),'cudnn':torch.backends.cuda.cudnn_sdp_enabled()}
 assert flags=={'flash':True,'math':False,'mem_efficient':False,'cudnn':False}
torch.cuda.synchronize();assert result.shape==a.shape and torch.isfinite(result).all().item() and torch.count_nonzero(result).item()==0
report={'schema':'third-node-pool-hardware-probe-v1','passed':True,'utc':datetime.now(timezone.utc).isoformat(),
 'package_sha256':pin,'host':socket.gethostname(),'job_id':os.environ['SLURM_JOB_ID'],'gpu':device.name,
 'gpu_total_bytes':device.total_memory,'cuda_capability':list(torch.cuda.get_device_capability(0)),
 'torch_cuda_version':torch.version.cuda,'flash_sdpa_flags':flags,'runtime_packages':versions,
 'nvidia_driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).splitlines(),
 'model_inference_calls':0,'human_annotation_calls':0,'natural_data_loaded':False,
 'scope':'Hardware and pinned environment admission only; no claim of full-model output parity or semantic validity'}
out.mkdir(exist_ok=False);raw=json.dumps(report,sort_keys=True,indent=2)+'\n'
(out/'result.json').write_text(raw);print(raw)

"""Admission for interchangeable Slurm compute A100 slots; never the login node."""
ALLOWED=('csis.cn1','csis.cn3','csis.cn4')
EXCLUDED=('csis.mn1','csis.cn2')
def acceptable_node(role,node):
    return node in ALLOWED if role=='gpu' else node==role

def gpu_flags(family):
    if family['allowed_nodes']!=list(ALLOWED):raise ValueError('allowed_node_policy_changed')
    return ['--partition=gpu-short','--nodes=1','--constraint=a100-80gb',
        '--exclude='+','.join(EXCLUDED),'--gres=gpu:a100-80gb:1','--cpus-per-task=4',
        '--mem='+str(family['host_memory_bytes']//1024**3)+'G','--time='+str(family['slurm_time_seconds']//60)]

def _probe():
    """Zero-model capability check inside the actual model allocation."""
    import os,socket,subprocess
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    if not os.environ.get('SLURM_JOB_ID','').isdigit() or socket.gethostname() not in ALLOWED:
        raise ValueError('approved_Slurm_compute_allocation_required')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise ValueError('one_BF16_GPU_required')
    device=torch.cuda.get_device_properties(0);cap=list(torch.cuda.get_device_capability(0))
    if 'A100' not in device.name or device.total_memory<75*1024**3 or cap!=[8,0]:
        raise ValueError('A10080GB_SM80_required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    zeros=torch.zeros((1,2,128,64),device='cuda',dtype=torch.bfloat16)
    with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output=torch.nn.functional.scaled_dot_product_attention(zeros,zeros,zeros,dropout_p=0.0)
        flags={'flash':torch.backends.cuda.flash_sdp_enabled(),'math':torch.backends.cuda.math_sdp_enabled(),
               'mem_efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),'cudnn':torch.backends.cuda.cudnn_sdp_enabled()}
        if flags!={'flash':True,'math':False,'mem_efficient':False,'cudnn':False}:raise ValueError('forced_flash')
    torch.cuda.synchronize()
    if output.shape!=zeros.shape or not torch.isfinite(output).all().item() or torch.count_nonzero(output).item()!=0:
        raise ValueError('analytic_zero_SDPA_fixture')
    result={'passed':True,'host':socket.gethostname(),'job_id':os.environ['SLURM_JOB_ID'],
        'gpu':device.name,'gpu_total_bytes':device.total_memory,'cuda_capability':cap,
        'torch_cuda_version':torch.version.cuda,'flash_sdpa_flags':flags,
        'driver_versions':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).splitlines(),
        'scope':'Hardware/software capability only, not full-model cross-device output equivalence',
        'model_inference_calls':0,'human_annotation_calls':0}
    del zeros,output
    torch.cuda.empty_cache()
    return result

def verify_allocated_gpu():
    # Isolate the capability fixture from the qualified loader's thread/RNG/CUDA state.
    import json,subprocess,sys
    from pathlib import Path
    child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--probe'],
        stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=90)
    if child.returncode!=0:raise RuntimeError('GPU capability subprocess failed: '+child.stderr[-2000:])
    if len(child.stdout)>16384:raise ValueError('capability_receipt_size')
    return json.loads(child.stdout)

if __name__=='__main__':
    import json,sys
    if sys.argv[1:]!=['--probe']:raise ValueError('probe_arguments')
    print(json.dumps(_probe(),sort_keys=True))

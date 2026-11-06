"""Characterize the pinned upstream mask semantics before novel-text inference."""
import os
from pathlib import Path
import sys

import pytest

torch=pytest.importorskip('torch')
SOURCE=Path(os.environ.get('SIGNSPARK_SOURCE','G:/Archive/Zenodo/SignSparK/source'))
if not (SOURCE/'signspark/rflow.py').is_file():
    pytest.skip('Pinned external SignSparK source is not installed',allow_module_level=True)
sys.path.insert(0,str(SOURCE))
os.environ['WANDB_MODE']='disabled'


def test_source_unet_mask_is_invariant_but_positive_observation_control_changes():
    from signspark.unet_model import Wilor_UNET
    torch.manual_seed(11)
    # Same source fusion path with reduced hidden width, avoiding 1.4B random
    # allocations; zero=False makes the observation positive control meaningful.
    model=Wilor_UNET(njoints=60,latent_dim=32,attention=False,zero=False,
                     text_conditioned=False,unet_out_mult=2,dataset='body').eval()
    x=torch.randn(1,60,16);t=torch.zeros(1)
    y={'mask':torch.ones(1,1,16,dtype=torch.bool),'lengths':torch.tensor([16])}
    empty=torch.zeros_like(x);unrelated=torch.randn_like(x)*3
    off=torch.zeros_like(x,dtype=torch.bool);on=~off
    with torch.no_grad():
        a=model(x,t,y=y,obs_x0=empty,obs_mask=off)
        b=model(x,t,y=y,obs_x0=unrelated,obs_mask=off)
        c=model(x,t,y=y,obs_x0=empty,obs_mask=on)
        d=model(x,t,y=y,obs_x0=unrelated,obs_mask=on)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.max(torch.abs(c-d)).item()>1e-5


def test_source_flow_initialization_and_each_step_ignore_masked_observations():
    from signspark.rflow import RFlow

    class UnitVelocity(torch.nn.Module):
        def forward(self,x,t,**kwargs):
            return torch.ones_like(x)

    torch.manual_seed(17)
    noise=torch.randn(1,60,16)
    off=torch.zeros_like(noise,dtype=torch.bool)
    flow=RFlow()
    results=[]
    for observed in (torch.zeros_like(noise),torch.randn_like(noise)*9):
        result,_=flow.decode(UnitVelocity(),noise=noise.clone(),keyframe_mask=off,
                            x_embed=observed,model_kwargs={'obs_mask':off,'obs_x0':observed},
                            ode_package='torchdiffeq',ode_stepnum=4)
        results.append(result)
    torch.testing.assert_close(results[0],noise+1,rtol=0,atol=1e-6)
    torch.testing.assert_close(results[0],results[1],rtol=0,atol=0)

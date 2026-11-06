"""Real single-GPU forward tests for the four ungated original Agent components."""
import argparse
import csv
import gc
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time
from construction_io import atomic_json, compile_program


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--component',choices=['all','deepseek','llava','owlv2','univtg'],default='all')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=Path(os.environ['AOTD_RUN_DIR'])
    agent=root/'agent_system'
    import torch
    import decord
    import numpy as np
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(agent/'configs/base_config.yaml')
    for name in cfg.load_models:cfg.load_models[name]=False
    for key,folder in [('deepseek','DeepSeek-Coder-6.7B-Instruct'),('llava','LLaVA-Video-7B-DPO'),('owlv2','OWLv2')]:
        cfg[key].model_path=str(root/'models/teachers'/folder)
    cfg.deepseek.prompt_path=str(root/'config/construction/program.prompt')
    cfg.llava.action_prompt_path=str(agent/'prompts/action_single.prompt')
    cfg.univtg.resume=str(root/'models/teachers/UniVTG/model_best.ckpt')
    cfg.univtg.clip_model_path=str(root/'models/teachers/CLIP/ViT-B-32.pt')
    cfg.univtg.save_dir=str(out/'univtg')
    path=out/'public_agent_config.yaml';OmegaConf.save(cfg,path)
    os.environ['AOTD_AGENT_CONFIG']=str(path)
    sys.path.insert(0,str(agent))
    import vision_models
    rows=list(csv.DictReader((root/'data/nextqa/train.csv').open()))
    row=next(x for x in rows if (root/'data/nextqa/NExTVideo'/f"{x['video']}.mp4").is_file())
    reader=decord.VideoReader(str(root/'data/nextqa/NExTVideo'/f"{row['video']}.mp4"),num_threads=1)
    frames=torch.from_numpy(reader.get_batch(np.linspace(0,len(reader),32,endpoint=False).astype(int)).asnumpy())
    result=dict(torch=torch.__version__,cuda=torch.version.cuda,video_id=row['video'],tests={})
    for name,cls in [('deepseek',vision_models.DeepSeekModel),('llava',vision_models.LLaVAVideoModel),('owlv2',vision_models.Owlv2Model),('univtg',vision_models.UniVTGModel)]:
        if a.component!='all' and name!=a.component:continue
        before=time.monotonic();print(f'[public Agent test] loading {name}',flush=True)
        previous=sys.argv;sys.argv=[previous[0]]
        try:model=cls(0)
        finally:sys.argv=previous
        loaded=time.monotonic()
        if name=='deepseek':
            answer=model.forward(row['question'],[row[f'a{i}'] for i in range(5)],input_type='video_clip, query, possible_answers')
            answer=answer.replace('video_clip, query, possible_answers, time_wait_between_lines, syntax','video_clip, query, possible_answers')
            compile_program(answer)
        elif name=='llava':answer=model.forward(frames,query=row['question'],possible_answers=[row[f'a{i}'] for i in range(5)])
        elif name=='owlv2':
            answer=model.forward(frames[0].permute(2,0,1),'person')
            answer=answer.tolist() if hasattr(answer,'tolist') else answer
        else:
            fps=float(reader.get_avg_fps())*32/len(reader)
            answer=model.forward(frames,fps,'a person is moving',len(reader)/(32*float(reader.get_avg_fps())))
            assert len(answer)==2 and answer[0]<=answer[1],answer
        torch.cuda.synchronize()
        result['tests'][name]=dict(status='PASS',load_seconds=loaded-before,forward_seconds=time.monotonic()-loaded,output=answer)
        atomic_json(out/'public_agent_validation.json',result)
        print(f'[public Agent test] {name} PASS: {str(answer)[:300]}',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'public_agent_validation.json',result)


if __name__=='__main__':main()

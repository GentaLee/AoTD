"""Offline GPU validation of both locally staged Llama teachers; no training data writes."""
import gc
import os
from pathlib import Path
import sys
import time
from construction_io import atomic_json


def main():
    root=Path(__file__).resolve().parents[1]
    output=Path(os.environ['AOTD_RUN_DIR'])
    from omegaconf import OmegaConf
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.manual_seed(20)
    agent=root/'agent_system'
    config=OmegaConf.load(agent/'configs/base_config.yaml')
    for key in config.load_models:config.load_models[key]=False
    config.llama.model_path=str(root/'models/teachers/Meta-Llama-3-8B-Instruct')
    config.select_answer_prompt_dir=str(agent/'prompts/select_answer.prompt')
    config.cot.save_cot_dir=str(output/'answer_selection_trace.txt')
    cfg=output/'llama_validation.yaml';OmegaConf.save(config,cfg)
    os.environ['AOTD_AGENT_CONFIG']=str(cfg)
    sys.path.insert(0,str(agent));sys.path.insert(0,str(root/'tools'))
    from vision_models import LLaMAModel
    import video_clip
    result=dict(status='RUNNING',fixture='isolated functionality checks; not training labels or CoT',tests={})
    before=time.monotonic();model=LLaMAModel(0)
    model.pipe.model.eval()
    # Exercise the actual select_answer tool, including non-dict observation input.
    video_clip.forward_=lambda _name,prompt,task: model.qa(prompt)
    answer=video_clip.select_answer('What color is the cup?', 'The visible cup is red.', ['red','blue'])
    assert answer.strip().strip('(). ')=='A',answer
    with torch.no_grad():
        inputs=model.tokenizer('The answer is',return_tensors='pt').to('cuda:0')
        logits=model.pipe.model(**inputs).logits
        assert torch.isfinite(logits).all().item()
    result['tests']['llama3']=dict(status='PASS',dtype=str(model.pipe.model.dtype),finite_logits=True,
                                  selected_answer=answer,seconds=time.monotonic()-before)
    atomic_json(output/'llama_validation.json',result)
    print('[local Llama GPU] Llama3 answer selection and finite BF16 logits PASS',flush=True)
    del logits,inputs,model;gc.collect();torch.cuda.empty_cache()
    from build_cot import TextBackend
    before=time.monotonic();backend=TextBackend('transfer')
    fixture=dict(question='What color is the cup?',code='def execute_command(video_clip, query, possible_answers):\n    return "A"',
                 execution_trace='call function Video_summary\nsummary result: The visible cup is red.',program_output='A')
    transferred=backend(fixture,None,300)
    assert transferred['rationale'].strip()
    del backend;gc.collect();torch.cuda.empty_cache()
    backend=TextBackend('filter')
    verdict=backend(fixture|transferred,None,300)
    assert verdict['filter_response'].strip(' .\n\t').casefold() in {'true','false'},verdict
    positive=backend(fixture|dict(rationale='First, inspect the visual observation: the cup is red. Second, compare this color with the options: option A is red and option B is blue. Therefore the observed red cup matches option A.'),None,300)
    negative=backend(fixture|dict(rationale='Unrelated words: banana, airplane, algebra.'),None,300)
    assert positive['accepted'] and not negative['accepted'],(positive,negative)
    result['tests']['llama31']=dict(status='PASS',transferred_rationale=transferred['rationale'],
                                   filter_response=verdict['filter_response'],accepted=verdict['accepted'],
                                   positive_control=positive,negative_control=negative,seconds=time.monotonic()-before)
    result['status']='PASS';atomic_json(output/'llama_validation.json',result)
    print('[local Llama GPU] Llama3.1 rationale transfer and filter PASS',flush=True)


if __name__=='__main__':main()

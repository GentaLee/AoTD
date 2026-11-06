"""Small real forward/generation checks while Taihua model transfer proceeds."""
import gc
import json
import os
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

root=Path(__file__).resolve().parents[1]
directory=root/'results'/('validation_local_llama_shanghai_'+time.strftime('%Y%m%dT%H%M%S',time.gmtime()))
directory.mkdir(parents=True,exist_ok=True)
targets=[('llama3',os.environ['AOTD_LLAMA3_SOURCE']),('llama31',os.environ['AOTD_LLAMA31_SOURCE'])]
report=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),tests={})
for name,path in targets:
    before=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,
                                            device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    prompt=tokenizer.apply_chat_template([dict(role='user',content='What is 2 + 2? Reply with the single digit only.')],tokenize=False,add_generation_prompt=True)
    batch=tokenizer(prompt,return_tensors='pt').to('cuda:0')
    with torch.inference_mode():
        logits=model(**batch).logits
        assert torch.isfinite(logits).all().item()
        ids=model.generate(**batch,max_new_tokens=16,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    answer=tokenizer.decode(ids[0,batch.input_ids.shape[1]:],skip_special_tokens=True).strip()
    assert answer=='4',answer
    report['tests'][name]=dict(status='PASS',answer=answer,finite_logits=True,dtype=str(model.dtype),seconds=time.monotonic()-before,source=path)
    (directory/'validation.json').write_text(json.dumps(report,indent=2))
    print(f'[Shanghai local model] {name} BF16 forward + generation PASS: {answer}',flush=True)
    del batch,ids,logits,model,tokenizer;gc.collect();torch.cuda.empty_cache()
report['status']='PASS';(directory/'validation.json').write_text(json.dumps(report,indent=2))
print('[Shanghai local model] Both PASS:',directory/'validation.json',flush=True)

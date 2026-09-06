"""Batch semantic filtering for materialized MolmoWeb SyntheticQA rows."""
from __future__ import annotations
import argparse, json
from pathlib import Path

PROMPT=(' /no_think\nJudge the screenshot QA pair. Do not provide analysis or <think>. '
        'Output one compact JSON object only: {"keep":true/false,"score":0-5}. '
        'Set keep=true only when the answer is grounded and correct; keep means score >= 3.')

def parse_json(s):
    a,b=s.find('{'),s.rfind('}')
    if a>=0 and b>a:
        try:return json.loads(s[a:b+1])
        except json.JSONDecodeError:pass
    return {'keep':False,'score':0,'reason':'unparseable'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--max-new-tokens',type=int,default=512); ap.add_argument('--start',type=int,default=0); ap.add_argument('--end',type=int,default=0); ap.add_argument('--audit',type=Path); a=ap.parse_args()
    from PIL import Image
    from vllm import LLM, SamplingParams
    rows=[json.loads(x) for x in a.input.open(encoding='utf-8')][a.start:a.end or None]
    llm=LLM(model=str(a.model),dtype='auto',max_model_len=4096,max_num_seqs=a.batch_size,limit_mm_per_prompt={'image':1},enforce_eager=True)
    sp=SamplingParams(temperature=0,max_tokens=a.max_new_tokens); a.output.parent.mkdir(parents=True,exist_ok=True); kept=0
    audit = (a.audit or a.output.with_suffix('.audit.jsonl')).open('w', encoding='utf-8')
    with a.output.open('w',encoding='utf-8') as out:
      for base in range(0,len(rows),a.batch_size):
        batch=rows[base:base+a.batch_size]; req=[]
        for r in batch:
          msg=r['messages'][0]; im=Image.open('/data1/datasets/xt_ama_adapter/filtered/molmoweb-syntheticqa-10k/'+r['image_path']).convert('RGB')
          text=PROMPT+'\nQuestion: '+msg['question']+'\nReference answer: '+msg['answer']
          req.append({'prompt':'<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'+text+'<|im_end|>\n<|im_start|>assistant\n','multi_modal_data':{'image':im}})
        for r,o in zip(batch,llm.generate(req,sp,use_tqdm=False)):
          raw=o.outputs[0].text; d=parse_json(raw); d.update({'sample_id':r.get('metadata',{}).get('id'), 'model_output':raw, 'question':r['messages'][0]['question'], 'answer':r['messages'][0]['answer'], 'image_path':r['image_path']})
          audit.write(json.dumps(d, ensure_ascii=False)+'\n')
          if bool(d.get('keep')) and float(d.get('score',0))>=3: kept+=1; out.write(json.dumps(d,ensure_ascii=False)+'\n')
        out.flush(); print(f'processed={min(base+a.batch_size,len(rows))} kept={kept}',flush=True)
    audit.close()
    print(json.dumps({'processed':len(rows),'kept':kept,'output':str(a.output)}))
if __name__=='__main__':main()

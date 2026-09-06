"""Batch semantic filtering of HumanTrajs with vLLM."""
from __future__ import annotations
import argparse, io, json
from pathlib import Path

PROMPT = ('Strictly review this web-agent trajectory for training a latent QA reader. '
          'Check action/screenshot consistency, progress toward the instruction, and '
          'whether it reaches a concrete completion. Return JSON only: '
          '{"keep":true/false,"score":0-5,"reason":"short"}. Keep score >= 3.')

def parse_json(text):
    a, b = text.find('{'), text.rfind('}')
    if a >= 0 and b > a:
        try: return json.loads(text[a:b+1])
        except json.JSONDecodeError: pass
    return {'keep': False, 'score': 0, 'reason': 'unparseable'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--parquet',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--max-new-tokens',type=int,default=64); ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image
    from vllm import LLM, SamplingParams
    rows=[json.loads(x) for x in a.manifest.open(encoding='utf-8')]
    if a.limit > 0: rows = rows[:a.limit]
    pf=pq.ParquetFile(a.parquet); cache={}
    def images_for(idx, names):
        start=0; rg=0
        while idx >= start+pf.metadata.row_group(rg).num_rows: start += pf.metadata.row_group(rg).num_rows; rg += 1
        if rg not in cache: cache[rg]=pf.read_row_group(rg,columns=['images']).column('images').to_pylist()
        raw=cache[rg][idx-start]; by={p:b for p,b in zip(rows_by_paths[idx],raw)}
        return [Image.open(io.BytesIO(bytes(by[n]))).convert('RGB') for n in names if n in by]
    # map source rows to image paths without loading all image bytes
    rows_by_paths={r['source_row']:r['image_paths'] for r in rows}
    llm=LLM(model=str(a.model),dtype='auto',max_model_len=8192,max_num_seqs=a.batch_size,
            limit_mm_per_prompt={'image':2}, enforce_eager=True)
    sp=SamplingParams(temperature=0,max_tokens=a.max_new_tokens)
    a.output.parent.mkdir(parents=True,exist_ok=True); kept=0
    with a.output.open('w',encoding='utf-8') as out:
      for base in range(0,len(rows),a.batch_size):
        batch=rows[base:base+a.batch_size]; req=[]
        for r in batch:
          steps=list(r['trajectory'].values()); ns=[s.get('screenshot') for s in steps if s.get('screenshot')]; picks=list(dict.fromkeys([ns[0],ns[-1]])); ims=images_for(r['source_row'],picks)
          text=PROMPT+'\nInstruction: '+json.dumps(r['instruction'],ensure_ascii=False)+'\nActions:\n'+'\n'.join(str(s.get('action',{}).get('action_str','')) for s in steps)
          image_tokens=''.join('<|vision_start|><|image_pad|><|vision_end|>' for _ in ims)
          prompt='<|im_start|>user\n'+image_tokens+text+'<|im_end|>\n<|im_start|>assistant\n'
          req.append({'prompt':prompt,'multi_modal_data':{'image':ims}})
        outs=llm.generate(req,sp, use_tqdm=False)
        for r,o in zip(batch,outs):
          raw=o.outputs[0].text; d=parse_json(raw); d.update({'source_row':r['source_row'],'sample_id':r['sample_id'],'model_output':raw})
          if bool(d.get('keep')) and float(d.get('score',0))>=3:
            kept+=1; d.update({'instruction':r['instruction'],'trajectory':r['trajectory'],'image_paths':r['image_paths']}); out.write(json.dumps(d,ensure_ascii=False)+'\n')
          out.flush()
        print(f'processed={min(base+a.batch_size,len(rows))} kept={kept}',flush=True)
    print(json.dumps({'processed':len(rows),'kept':kept,'output':str(a.output)}))
if __name__=='__main__': main()

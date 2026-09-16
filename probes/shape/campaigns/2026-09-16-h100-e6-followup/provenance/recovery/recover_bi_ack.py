import json,hashlib,importlib.util,sys,time
from pathlib import Path
p=Path('/root/c2-evidence/probes/shape/next_campaign/run_followup.py')
spec=importlib.util.spec_from_file_location('runner',p);m=importlib.util.module_from_spec(spec);sys.modules['runner']=m;spec.loader.exec_module(m)
r=Path('/root/c2-campaign/results');a=Path('/root/c2-campaign/acks/BI_b.json');manifest=r/'handoff/BI_b/manifest.json'
m.verify_ack(a,'BI_b',manifest)
for row in m.load_json(manifest)['files']:
 f=r/row['path'];assert f.stat().st_size==row['size'];assert m.sha256_file(f)==row['sha256'],str(f)
state=m.load_state(r);assert state['completed_labels']==[]
m.verify_runner_copy(r,state,p)
for arm in state['records']: m.verify_immutable_record(r,state,arm)
summary=m.validate_replay_summary(r,'Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0_batch_invariant','replay_cold_followup_b')
assert m.load_json(summary)['requirements_met'] is True
recovery=Path('/root/c2-campaign/recovery');recovery.mkdir(exist_ok=True)
before=(r/'runner_state.json').read_bytes();(recovery/'runner_state_before_bi_ack.json').write_bytes(before)
state['completed_labels'].append('BI_b');m.save_state(r,state)
after=(r/'runner_state.json').read_bytes();(recovery/'runner_state_after_bi_ack.json').write_bytes(after)
m.write_json(recovery/'BI_b_ack_recovery.json',{'reason':'Operator ACK was non-atomically written; runner read empty/partial JSON and exited after successful arm and durable verified transfer. Complete ACK and every handoff file verified before reconciling completed label. Future ACKs use atomic rename.','label':'BI_b','at_epoch':time.time(),'handoff_manifest_sha256':m.load_json(manifest)['sha256'],'before_state_sha256':hashlib.sha256(before).hexdigest(),'after_state_sha256':hashlib.sha256(after).hexdigest(),'gpu_work_repeated':False})
print('BI_b ACK recovered with all handoff files and immutable records verified')

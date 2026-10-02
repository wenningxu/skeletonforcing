"""Fixed OF004 progression under the user's small-experiment authorization."""
import datetime,json,os,subprocess,sys
from pathlib import Path
from .overfit import approval_check


def main():
    out=Path('/workspace/outputs'); record=out/'OF004-suite.json'
    if record.exists(): raise FileExistsError('Never relaunch a previous suite')
    for arm in ['a','b']:
        approval_check(f'approvals/of004-{arm}.json',f'configs/overfit_wan_v1_{arm}.json')
    status={'stage':'starting','a':None,'b':None}
    def save(): record.write_text(json.dumps(status,indent=2)+'\n')
    save()
    for arm in ['a','b']:
        if arm=='b':
            a=json.loads((out/'OF004-A/summary.json').read_text())
            passed=all(a[group][metric] for group in ['trained_layout_trained_angle','trained_layout_unseen_angle']
                for metric in ['reconstruction_passed','response_passed'])
            deadline=datetime.datetime.fromisoformat(os.environ['EXPERIMENT_DEADLINE_UTC'].replace('Z','+00:00'))
            remaining=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            if not passed or remaining<35*60:
                status['b']={'status':'SKIPPED','reason':'single-clip gates failed' if not passed else 'less than 35 minutes remain'}
                status['stage']='finished'; save(); print(json.dumps(status),flush=True); return
        status['stage']=f'running_{arm}'; save()
        result=subprocess.run([sys.executable,'-m','motion_valley.control_experiment',
            '--config',f'configs/overfit_wan_v1_{arm}.json','--approval-file',f'approvals/of004-{arm}.json',
            '--data','/workspace/data','--output',str(out/f'OF004-{arm.upper()}')])
        status[arm]={'returncode':result.returncode,'status':'COMPLETED' if result.returncode==0 else 'FAILED'}
        if result.returncode:
            status['stage']='failed';
            if arm=='a': status['b']={'status':'SKIPPED','reason':'first program failed or incomplete'}
            save(); raise SystemExit(result.returncode)
        save()
    status['stage']='finished'; save(); print(json.dumps(status),flush=True)


if __name__=='__main__': main()

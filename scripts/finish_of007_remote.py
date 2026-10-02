"""Keep one SSH connection for monitoring and evidence retrieval; never launch training."""
import datetime,hashlib,json,time
from pathlib import Path
import paramiko

root=Path(__file__).resolve().parents[1]
info=json.loads((root/'reports/pod_connection.json').read_text())
resource=json.loads((root/'reports/resources_of007.json').read_text())
assert info['pod_id']==resource['pod_id'] and resource['experiment']=='OF007' and not info['stale']
deadline=datetime.datetime.fromisoformat(resource['hard_deadline_utc'].replace('Z','+00:00')).timestamp()-90
key=paramiko.Ed25519Key.from_private_key_file(str(root/'.secrets/runpod_ed25519'),password='""')
last='';done=False
for attempt in range(1,7):
    if time.time()>=deadline:break
    client=paramiko.SSHClient(); known=root/'.secrets/known_hosts'
    if known.exists():client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(info['host'],port=info['port'],username=info['username'],pkey=key,
            look_for_keys=False,allow_agent=False,timeout=20,banner_timeout=20,auth_timeout=20)
        client.save_host_keys(str(known));client.get_transport().set_keepalive(10)
        print('Monitoring connection established',flush=True)
        def execute(command):
            _,out,err=client.exec_command(command,timeout=55)
            text=out.read().decode(); error=err.read().decode(); status=out.channel.recv_exit_status()
            if status:raise RuntimeError(f'Remote read/collection failed: {error}')
            return text
        while time.time()<deadline:
            status=execute("tail -n 2 /workspace/outputs/OF007-run.log; if test -f /workspace/outputs/OF007/SUITE_COMPLETED && test -f /workspace/outputs/OF007-pip_freeze.txt; then printf '\\nEVIDENCE_READY\\n'; fi")
            if status!=last:print(status,flush=True);last=status
            if 'EVIDENCE_READY' in status:
                code=(root/'scripts/collect_of007_evidence.py').read_text()
                assert '\nOF007_COLLECT_END\n' not in code
                collection=json.loads(execute("python3 - <<'OF007_COLLECT_END'\n"+code+"\nOF007_COLLECT_END\n"))
                print(json.dumps(collection),flush=True)
                (root/'reports/of007_collection.json').write_text(json.dumps(collection,indent=2))
                dest=root/'outputs/OF007-evidence.tar.gz'; temp=dest.with_suffix('.gz.partial')
                with client.open_sftp() as sftp:sftp.get(collection['path'],str(temp),max_concurrent_prefetch_requests=64)
                if hashlib.sha256(temp.read_bytes()).hexdigest()!=collection['sha256']:raise ValueError('Download hash mismatch')
                temp.replace(dest); print('Evidence downloaded and archive hash verified',flush=True)
                done=True;break
            time.sleep(30)
        if done:break
    except Exception as e:
        print(f'Connection/collection attempt {attempt}: {type(e).__name__}: {e}',flush=True)
        time.sleep(5)
    finally:client.close()
if not done:raise RuntimeError('Evidence not collected; preserve volume and release pod before deadline')

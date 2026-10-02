"""Single-shot upload/launch; a failed/uncertain launch MUST be inspected, not retried."""
import datetime,json
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parents[1]
resource=json.loads((root/'reports/resources_of009.json').read_text())
info=json.loads((root/'reports/pod_connection.json').read_text())
assert info['pod_id']==resource['pod_id'] and resource['experiment']=='OF009' and resource['status']=='ACTIVE' and not info['stale']
client=paramiko.SSHClient();known=root/'.secrets/known_hosts'
if known.exists():client.load_host_keys(str(known))
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
key=paramiko.Ed25519Key.from_private_key_file(str(root/'.secrets/runpod_ed25519'),password='""')
client.connect(info['host'],port=info['port'],username=info['username'],pkey=key,
               look_for_keys=False,allow_agent=False,timeout=25,banner_timeout=25,auth_timeout=25)
client.save_host_keys(str(known));client.get_transport().set_keepalive(10)
with client.open_sftp() as sftp:
    sftp.put(str(root/'outputs/OF009-source.tar.gz'),'/workspace/outputs/OF009-source.tar.gz')
print('Source uploaded',flush=True)
deadline=resource['hard_deadline_utc'];digest=resource['source_archive_sha256']
command=f'''set -eu
echo '{digest}  /workspace/outputs/OF009-source.tar.gz' | sha256sum -c -
test ! -e /workspace/outputs/OF009
mkdir /workspace/outputs/OF009-launch-lock
tar xzf /workspace/outputs/OF009-source.tar.gz -C /workspace/motion-valley
cd /workspace/motion-valley
seconds=$(( $(date -d '{deadline}' +%s) - $(date +%s) - 60 ))
test "$seconds" -gt 900
setsid timeout --signal=TERM --kill-after=30s "$seconds" env EXPERIMENT_DEADLINE_UTC='{deadline}' bash scripts/run_of009.sh > /workspace/outputs/OF009-run.log 2>&1 < /dev/null &
pid=$!
echo "$pid" > /workspace/outputs/OF009-launch-lock/pid
echo "OF009_SINGLE_LAUNCH_PID=$pid"
'''
_,out,err=client.exec_command(command,timeout=55)
response=out.read().decode();error=err.read().decode();status=out.channel.recv_exit_status()
print(response,flush=True);print(error,flush=True)
client.close()
if status:raise RuntimeError('Launch failed; inspect lock/log before any further action')
resource['launch_response']=response;resource['live_stage']='launched once; dependency setup and tests'
(root/'reports/resources_of009.json').write_text(json.dumps(resource,indent=2)+'\n')

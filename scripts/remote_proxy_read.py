"""Read-only log fallback through the pod's API-returned SSH proxy."""
from pathlib import Path
import time
import paramiko

root=Path(__file__).resolve().parents[1]
client=paramiko.SSHClient()
known=root/'.secrets/known_hosts'
if known.exists():client.load_host_keys(str(known))
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
key=paramiko.Ed25519Key.from_private_key_file(str(root/'.secrets/runpod_ed25519'),password='""')
client.connect('ssh.runpod.io',port=22,username='r304ca6vtjiyit-64411e2f',pkey=key,
    look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20,banner_timeout=20)
client.save_host_keys(str(known))
channel=client.invoke_shell(); channel.settimeout(5)
channel.send("tail -n 3 /workspace/outputs/OF006-inpaint-run.log\nexit\n")
deadline=time.monotonic()+40
while time.monotonic()<deadline:
    if channel.recv_ready():
        block=channel.recv(65536)
        if not block:break
        print(block.decode(errors='replace'),end='',flush=True)
    elif channel.exit_status_ready():break
    else:time.sleep(.2)
client.close()

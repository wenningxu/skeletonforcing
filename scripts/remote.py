"""SSH/SFTP helper; .secrets is never included in uploaded project bundles."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import paramiko

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['exec','upload','download'])
    ap.add_argument('value',nargs='?',default='')
    ap.add_argument('--to',default='/workspace/motion-valley')
    args=ap.parse_args()
    info=json.loads((ROOT/'reports/pod_connection.json').read_text())
    client=paramiko.SSHClient()
    known=ROOT/'.secrets/known_hosts'
    if known.exists(): client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key=paramiko.Ed25519Key.from_private_key_file(str(ROOT/'.secrets/runpod_ed25519'),password='""')
    client.connect(info['host'],port=info['port'],username=info['username'],
                   pkey=key,look_for_keys=False,allow_agent=False,timeout=25)
    client.save_host_keys(str(known))
    if args.action=='exec':
        _,out,err=client.exec_command(args.value,timeout=55)
        print(out.read().decode(),end=''); print(err.read().decode(),end='',file=sys.stderr)
        code=out.channel.recv_exit_status(); client.close(); sys.exit(code)
    sftp=client.open_sftp()
    if args.action=='upload':
        archive=ROOT/'outputs/code.tar.gz'; archive.parent.mkdir(exist_ok=True)
        with tarfile.open(archive,'w:gz') as z:
            for folder in ['motion_valley','scripts','tests','configs','docs','third_party','approvals','prepared']:
                for p in (ROOT/folder).rglob('*'):
                    if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts:
                        z.add(p,arcname=p.relative_to(ROOT))
            z.add(ROOT/'requirements.txt',arcname='requirements.txt')
            for report in (ROOT/'reports').glob('approval_report*_zh.md'):
                z.add(report,arcname=str(report.relative_to(ROOT)))
        sftp.put(str(archive),'/workspace/motion-valley-code.tar.gz')
        # destination is a fixed task-owned directory, not user-provided shell text
        _,out,err=client.exec_command('mkdir -p /workspace/motion-valley && tar xzf /workspace/motion-valley-code.tar.gz -C /workspace/motion-valley')
        code=out.channel.recv_exit_status(); print(err.read().decode());
        if code: raise RuntimeError('remote extraction failed')
        print('Uploaded',archive.stat().st_size,'bytes')
    else:
        dest=Path(args.to); dest.parent.mkdir(parents=True,exist_ok=True)
        temporary=dest.with_suffix(dest.suffix+'.partial')
        sftp.get(args.value,str(temporary),max_concurrent_prefetch_requests=64)
        temporary.replace(dest); print(dest)
    client.close()


if __name__=='__main__': main()

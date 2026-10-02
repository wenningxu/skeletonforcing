"""Download publisher assets; extract only motion, text, splits and evaluators.

Archives are SHA256-recorded, HF revision is pinned in a provenance manifest.
No pretrained generator or latent tokens are required by this project.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from concurrent.futures import ThreadPoolExecutor
import threading
import zlib
from huggingface_hub import HfApi, hf_hub_download


def wanted(name, archive):
    p = name.replace('\\','/')
    if archive == 'deps.zip':
        return any('/'+part+'/' in '/'+p for part in ('t2m','glove','t5_umt5-xxl-enc-bf16'))
    parts = p.split('/')
    if any(x in parts for x in ('new_joint_vecs','motions','texts')):
        return p.endswith(('.npy','.txt'))
    return len(parts) <= 2 and p.endswith(('.txt','.npy','.json'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--datasets',nargs='+',default=['HumanML3D','BABEL_streamed'])
    ap.add_argument('--revision',default=None)
    args=ap.parse_args(); args.root.mkdir(parents=True,exist_ok=True)
    repo='ShandaAI/FloodDiffusionDownloads'
    info=HfApi().model_info(repo,revision=args.revision,files_metadata=True)
    manifest={'repo':repo,'revision':info.sha,'files':[]}
    for filename in ['deps.zip']+[x+'.zip' for x in args.datasets]:
        print('Downloading',filename,flush=True)
        path=Path(hf_hub_download(repo,filename,revision=info.sha,cache_dir=args.root/'hf-cache'))
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        record=next(x for x in info.siblings if x.rfilename==filename)
        if record.lfs is not None and digest!=record.lfs.sha256:
            raise ValueError('Archive SHA256 differs from publisher')
        target=args.root if filename=='deps.zip' else args.root/'raw_data'
        target.mkdir(exist_ok=True)
        count=0; size=0
        with zipfile.ZipFile(path) as z:
            items=[item for item in z.infolist() if not item.is_dir() and wanted(item.filename,filename)]
            local=threading.local()
            readers=[]
            def extract(item):
                if not hasattr(local,'archive'):
                    local.archive=zipfile.ZipFile(path)
                    readers.append(local.archive)
                out=(target/item.filename).resolve()
                if not out.is_relative_to(target.resolve()):
                    raise ValueError('unsafe archive member')
                valid=out.exists() and out.stat().st_size==item.file_size
                if valid and filename!='deps.zip':
                    valid=(zlib.crc32(out.read_bytes()) & 0xffffffff)==item.CRC
                if not valid:
                    out.parent.mkdir(parents=True,exist_ok=True)
                    temporary=out.with_suffix(out.suffix+'.extracting')
                    with local.archive.open(item.filename) as src,temporary.open('wb') as dst:
                        import shutil
                        shutil.copyfileobj(src,dst,1024*1024)
                    temporary.replace(out)
                return item.file_size
            # Parallel filesystem operations avoid serial network-volume latency.
            # Separate reader per worker avoids sharing a buffered network file.
            with ThreadPoolExecutor(max_workers=16) as workers:
                for extracted_size in workers.map(extract,items):
                    count+=1; size+=extracted_size
                    if count%5000==0: print('Extracted so far',count,flush=True)
            for reader in readers: reader.close()
        manifest['files'].append(dict(file=filename,sha256=digest,extracted_files=count,extracted_bytes=size))
        (args.root/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
        print('Extracted',count,'files;',size,'bytes',flush=True)


if __name__=='__main__': main()

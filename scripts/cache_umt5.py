import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motion_valley.text_umt5 import cache_texts

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--captions',required=True,help='JSON list of original caption strings')
    p.add_argument('--deps',required=True); p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.device.startswith('cuda'):
        raise SystemExit('GPU cache creation must be invoked by the approved experiment runner; standalone GPU launch disabled')
    cache_texts(json.loads(Path(args.captions).read_text()),args.deps,args.output,args.device)

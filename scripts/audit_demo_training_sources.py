"""Audit prepared demo membership against the frozen official annotations."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
ANNOTATIONS=Path('F:/model_weigth/data/csl-daily')
HASHES=dict(train='ef77f56e517cf5ecf16a7bf8035de1965975164b92a4ca3348c920912365481f',
            dev='f14660c46f9714425db2de33d0f037a8ed19cf945fd70ae9e0cde2efbd62bd4a',
            test='ebbbb48a3ff4430106eac39977759398a82afe2f448ca6378a0012ade4972458')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def normal(text):return ''.join(unicodedata.normalize('NFKC',text).split())

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    annotations={}
    for split,digest in HASHES.items():
        path=ANNOTATIONS/f'csl-daily.{split}'
        assert sha(path)==digest  # Only then open the already-trusted official pickle.
        with gzip.open(path,'rb') as f:annotations[split]={r['name']:r for r in pickle.load(f)}
    catalog=read(a.root/'catalog.json');build=read(a.root/'build-record.json')
    assert build['source_manifest_sha256']==sha(ROOT/'artifacts/train-application-validation-v2/manifest.json')
    rows=[]
    for item in catalog['items']:
        name=item['sample_id'];assert name in annotations['train']
        assert name not in annotations['dev'] and name not in annotations['test']
        assert normal(item['annotation'])==normal(annotations['train'][name]['text'])
        assert sha(a.root/item['media'])==item['sha256']
        source=next(r for r in build['items'] if r['id']==item['id'])
        assert sha(source['source_video'])==source['source_sha256']
        rows.append(dict(id=item['id'],sample_id=name,official_train=True,absent_dev_test=True,
                         annotation_matches=True,media_sha256=item['sha256'],tail_seconds=source['tail_seconds']))
    report=dict(catalog_sha256=sha(a.root/'catalog.json'),annotation_sha256=HASHES,items=rows,
                scope='source_membership_and_annotation_not_human_semantic_or_imitation_acceptance')
    (a.root/'source-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'audited':len(rows),'all_train':True,'all_annotations_match':True}))

if __name__=='__main__':main()

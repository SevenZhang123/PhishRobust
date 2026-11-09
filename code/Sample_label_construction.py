# A sample-label mapping txt file was constructed based on legitimate website datasets
# and phishing website datasets to facilitate PhishRobust's data reading and effectiveness evaluation.

import random
from pathlib import Path

BASE_DIR   = Path("../data")
BENIGN_ROOT = BASE_DIR / 'benign_samples_test'
PHISH_ROOT  = BASE_DIR / 'phish_samples_test'

def collect_samples(root: Path, label: int):
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        shot = sub / 'shot.png'
        url_txt = sub / 'url.txt'
        if not shot.exists() or not url_txt.exists():
            continue
        rel_path = shot.relative_to(BASE_DIR).as_posix()
        url = url_txt.read_text(encoding='utf-8').strip()
        yield rel_path, url, label

with open(BASE_DIR / 'sample_label.txt', 'w', encoding='utf-8') as f:
    for path, url, label in collect_samples(BENIGN_ROOT, 0):
        f.write(f'{path}\t{url}\t{label}\n')
    for path, url, label in collect_samples(PHISH_ROOT, 1):
        f.write(f'{path}\t{url}\t{label}\n')

print('sample_label.txt has been generated!')
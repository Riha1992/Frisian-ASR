import sys
import os

"""
Example:
$ python check_raw_data.py path/to/data/cv-corpus-16.1-2023-12-06/fy-NL
"""

data_dir = sys.argv[1]

durations = {}

with open(os.path.join(data_dir, 'clip_durations.tsv')) as f:
    next(f)
    for line in f:
        line = line.strip().split('\t')
        clip = line[0]
        duration = float(line[1])
        durations[clip] = duration


for split in ['train', 'dev', 'test', 'validated']:
    duration = 0
    with open(os.path.join(data_dir, f'{split}.tsv')) as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            clip = line[1]
            if clip not in durations:
                print(f'WARNING Clip {clip} not found in clip_durations.tsv ({split}.tsv)')
                sys.exit(1)
            duration += durations[clip]

    h = int(duration // 3600_000)
    m = int((duration % 3600) // 60_000)
    s = int(duration % 60_000 // 1000)
    print(f"{split:>15}: {h:02d}:{m:02d}:{s:02d}")
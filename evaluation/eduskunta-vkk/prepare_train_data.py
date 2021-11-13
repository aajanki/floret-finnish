import pandas as pd
from pathlib import Path

data_dir = Path('evaluation/data/eduskunta-vkk')

df = pd.read_csv(data_dir / 'train.csv.bz2')
with open(data_dir / 'train.txt', 'w') as f:
    f.write('\n'.join(df['sentence']))


df = pd.read_csv(data_dir / 'dev.csv.bz2')
with open(data_dir / 'dev.txt', 'w') as f:
    f.write('\n'.join(df['sentence']))


df = pd.read_csv(data_dir / 'test.csv.bz2')
with open(data_dir / 'test.txt', 'w') as f:
    f.write('\n'.join(df['sentence']))

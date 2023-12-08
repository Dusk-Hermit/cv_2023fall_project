import os
import sys

old_pt_path='/project/train/models/0_error.pt'

with open(old_pt_path, 'r') as f:
    print(f.read())

l=os.listdir('/project/train/models')
for x in l:
    print(x)

print('Done')
import os
for d in sorted(os.listdir('Dataset')):
    p = os.path.join('Dataset', d)
    if os.path.isdir(p):
        print(f"{d}: {len(os.listdir(p))}")

import os
import pickle
output = ['clip,seizure,early']
files = [i for i in os.listdir('.') if i.endswith('.pkl')]
files.sort()
for filename in files:
    with open(filename) as f:
        p = pickle.load(f)
        for l, s in zip(p[0], p[1]):
            output.append('%s,%f,%f' % (l,s,s))
print('\n'.join(output))

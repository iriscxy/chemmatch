import json
import pdb
from collections import Counter
import os

name='ver1'
if not os.path.exists('data/set1_{}/'.format(name)):
    os.mkdir('data/set1_{}/'.format(name))
fw=open('data/set1_{}/test_output.json'.format(name),'w')
fw2=open('data/set1_{}/dev_output.json'.format(name),'w')
f=open('test.json')
all_lines=f.readlines()
idx=0
keys=[]
all_content=dict()
for line in all_lines:
    ori_content=json.loads(line)
    content=dict()
    content['ori'] = ori_content['title'] + ' [SEP] ' + ori_content['abstract']
    if ori_content['answer'] == 'no':
        content['label'] = 0
    elif ori_content['answer'] == 'yes':
        content['label'] = 1
    elif ori_content['answer'] == 'maybe':
        content['label'] = 2
    keys.append(ori_content['answer'])
    all_content[str(idx)] = content
    idx += 1


value_counts = Counter(keys)
for value, count in value_counts.items():
    print(f"Value '{value}' occurs {count} time(s).")


json.dump(all_content,fw, indent=4,ensure_ascii=False)
json.dump(all_content,fw2, indent=4,ensure_ascii=False)
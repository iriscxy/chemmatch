import json
import pdb
from collections import Counter
import os

name='ver1'
if not os.path.exists('data/set1_{}/'.format(name)):
    os.mkdir('data/set1_{}/'.format(name))
fw=open('data/set1_{}/train_label_output.json'.format(name),'w')
idx=0
keys=[]
all_content=dict()

f=open('train_label.json')
lines=f.readlines()
for line in lines:
    ori_content=json.loads(line)
    content=dict()
    idx+=1
    content['ori']=ori_content['title']+' [SEP] '+ori_content['abstract']
    if 'answer' in ori_content.keys():
        keys.append(ori_content['answer'])
        if ori_content['answer']=='no':
            content['label']=0
        elif ori_content['answer']=='yes':
            content['label']=1
        elif ori_content['answer']=='maybe':
            content['label']=2
    all_content[str(idx)]=content
value_counts = Counter(keys)
for value, count in value_counts.items():
    print(f"Value '{value}' occurs {count} time(s).")


json.dump(all_content,fw, indent=4)
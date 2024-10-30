import json
import pdb
import os
name='ver1'

if not os.path.exists('data/set1_{}/'.format(name)):
    os.mkdir('data/set1_{}/'.format(name))
fw=open('data/set1_{}/train_unlabel_output.json'.format(name),'w')
f=open('train_unlabel.json')
all_lines=f.readlines()
idx=0
all_content=dict()
for line in all_lines:
    ori_content=json.loads(line)
    content=dict()
    idx+=1

    content['ori']=ori_content['title']+' [SEP] '+ori_content['abstract']
    content['aug_0']=ori_content['title']+' [SEP] '+ori_content['aug_german']
    content['aug_1']=ori_content['aug_german_title']+' [SEP] '+ori_content['abstract']
    all_content[str(idx)]=content

json.dump(all_content,fw, indent=4,ensure_ascii=False)
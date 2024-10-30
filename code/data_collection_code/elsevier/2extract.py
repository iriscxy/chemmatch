# from climb_elsevier_xml to output_elsevier.json
import json
import os
import re
from bs4 import BeautifulSoup

from tqdm import tqdm
import pdb

doi_set = set()
cnt = 0


def clean(abstract):
    abstract = abstract.strip()
    if abstract.lower().startswith('abstract'):
        abstract = abstract[abstract.lower().index('abstract') + 8:]
    abstract = abstract.replace('\n', ' ')
    pattern = r'\<.*?\>'
    abstract = re.sub(pattern, '', abstract)
    abstract = re.sub(' +', ' ', abstract)
    abstract = re.sub('\t+', ' ', abstract)
    return abstract


fw1 = open('final.json', 'w')
with open('climb_elsevier.json') as file2:
    print('elsevier')
    for line in file2:
        content = json.loads(line)
        doi = content['DOI']
        data = content['collect_text']
        jounal_name = content['container-title']
        bs = BeautifulSoup(data, 'lxml')
        try:
            # abstract = bs.find_all("ce:simple-para")[0].text
            abstract = bs.find_all("dc:description")[0].text
            abstract = clean(abstract)
            print(abstract)
            if doi not in doi_set:
                doi_set.add(doi)

                cnt += 1
                print(str(cnt))
                if len(abstract.split()) < 20:
                    print('too short')
                new_content = dict()
                new_content['title'] = content['title'][0]
                new_content['abstract'] = abstract
                new_content['time'] = content['created']['date-parts'][0]
                new_content['DOI'] = doi
                new_content['jounal_name'] = jounal_name
                new_content['subject'] = content['subject']
                json.dump(new_content, fw1)
                fw1.write('\n')
        except:
            print('wrong')

            continue

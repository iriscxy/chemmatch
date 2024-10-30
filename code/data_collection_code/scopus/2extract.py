import json
import os
import pdb
import re
from tqdm import tqdm
keywords = ['catalyst', 'catalysis', 'catalysts', 'catalytic', 'catalyzed']


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

def extract_date(date_string):
    year, month, day = date_string.split('-')
    return [int(year), int(month), int(day)]
#
doi_set=set()
id_set=set()
fw=open('final.json','w')
for keyword in keywords:
    print(keyword)
    if os.path.exists('scopus/{}.txt'.format(keyword)):
        f = open('scopus/{}.txt'.format(keyword))
        lines = f.readlines()
        for line in tqdm(lines):
            content = json.loads(line)
            id = content['dc:identifier']
            title = content['dc:title']
            try:
                keywords_list = content['authkeywords'].split(' | ')
            except:
                keywords_list=[]
            time = extract_date(content['prism:coverDate'])
            if 'dc:description' in content.keys():
                abstract = clean(content['dc:description'])
                try:
                    journal_name = content['prism:publicationName']
                except:
                    journal_name='empty'
                if  len(abstract.split()) > 20:
                    if 'prism:doi' in content.keys():
                        doi = content['prism:doi']
                        if doi not in doi_set:
                            doi_set.add(doi)
                            content = {}
                            content['title'] = title
                            content['abstract'] = abstract
                            content['DOI'] = doi
                            content['time'] = time
                            content['scopus_ids'] = id
                            content['journal_name']=journal_name
                            content['subject']=keywords_list
                            json.dump(content, fw, ensure_ascii=False)
                            fw.write('\n')
                    elif id not in id_set:
                        content = {}
                        content['title'] = title
                        content['abstract'] = abstract
                        content['source'] = 'scopus'
                        content['time'] = time
                        content['scopus_ids'] = id
                        content['journal_name'] = journal_name
                        content['subject'] = keywords_list
                        json.dump(content, fw, ensure_ascii=False)
                        fw.write('\n')
                    id_set.add(id)

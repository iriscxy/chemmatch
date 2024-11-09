import fileinput
import json
import os
import requests


cnt = 0
publisher_set = set()
already_set = set()

f = open('all_dois.txt')
lines = f.readlines()
lines = [line.strip() for line in lines]
for line in lines:
    already_set.add(line)
f.close()

journal_name = 'elsevier'
if os.path.exists('crawl_{}.json'.format(journal_name)):
    f = open('crawl_{}.json'.format(journal_name))
    lines = f.readlines()
    for line in lines:
        content = json.loads(line)
        doi = content['DOI']
        already_set.add(doi)
    f.close()

with open('cross_byjournal/elsevier/crawl_elsevier_xml.json') as file2:
    for line in file2:
        content = json.loads(line)
        doi = content['DOI']
        already_set.add(doi)

with open('crawl_{}.json'.format(journal_name)) as file2:
    for line in file2:
        content = json.loads(line)
        doi = content['DOI']
        already_set.add(doi)

def notbiological(abstract):
    if abstract.find('DNA') != -1 or abstract.find('RNA') != -1:
        return False
    lower_line = abstract.lower()
    if lower_line.find('proteins') == -1 and lower_line.find('enzym') == -1 and lower_line.find(
            ' biological catal') and lower_line.find('organic catal') == -1 and lower_line.find(
        'microbial fuel cell'):
        return True
    else:
        return False

fw = open('crawl_{}.json'.format(journal_name), 'a')
with open('all_journal_byword.json') as file2:
    for line in file2:
        content = json.loads(line)
        doi = content['DOI']
        if doi in already_set:
            continue
        publisher = content['publisher']

        if 'link' in content.keys():
            link = content['link']
            url = link[0]['URL']
            title = content['title'][0]
            if 'elsevier' in url:
                number = content['alternative-id'][0]
                url='https://api.elsevier.com/content/article/pii/{}?apiKey=&httpAccept=application%2Fxml&insttoken='.format(number)

                rr = requests.get(url)
                # data=ElsevierSoup.parse(rr.text)
                if not notbiological(title):
                    continue
                print(title)
                try:
                    content['collect_text'] = rr.text
                except:
                    print('no abstract for {}'.format(doi))
                json.dump(content, fw)
                fw.write('\n')
print(str(cnt))

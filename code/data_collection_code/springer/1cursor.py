#from all_journal_byword3 to climb_springer.json

import fileinput
import json
import os
import pdb
import re
from requests_html import HTMLSession
import requests
from bs4 import BeautifulSoup

session = HTMLSession()
cnt = 0
publisher_set = set()
already_set = set()

f = open('all_dois.txt')
lines = f.readlines()
lines = [line.strip() for line in lines]
for line in lines:
    already_set.add(line)
f.close()
journal_name = 'springer'
if os.path.exists('climb_{}.json'.format(journal_name)):
    f = open('climb_{}.json'.format(journal_name))
    lines = f.readlines()
    for line in lines:
        content = json.loads(line)
        doi = content['DOI']
        already_set.add(doi)
    f.close()

fw = open('climb_{}.json'.format(journal_name), 'a')
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
            title = content['title']
            if 'springer' in url:
                print (title)
                if url.endswith('.pdf'):
                    url=url.replace('content/pdf','article')[:-4]
                if 'content/pdf' in url:
                    url=url.replace('content/pdf','article')
                if 'index/pdf'in url:
                    url=url.replace('index/pdf','article')
                    url=url.split('/')
                    url[2]='link.springer.com'
                    url='/'.join(url)

                rr = requests.get(url)
                bs = BeautifulSoup(rr.content)
                for div in bs.find_all("h3", {'id': 'FPar1'}): div.decompose()
                for div in bs.find_all("section", {'data-title': 'Acknowledgements'}): div.decompose()
                for div in bs.find_all("section", {'aria-labelledby': 'author-information'}): div.decompose()
                for div in bs.find_all("li", {
                    'class': 'c-article-references__item js-c-reading-companion-references-item'}): div.decompose()
                for div in bs.find_all("div", {'class': 'app-footer__container'}): div.decompose()
                for div in bs.find_all("p", {'id': re.compile(r'ref/*')}): div.decompose()
                for div in bs.find_all("a", {'data-track': "click"}): div.decompose()
                for div in bs.find_all("meta", {'name': 'citation_reference'}): div.decompose()
                for div in bs.find_all("section", {'data-title': 'Additional information'}): div.decompose()
                for div in bs.find_all("p", {
                    'class': 'c-bibliographic-information__download-citation u-hide-print'}): div.decompose()
                for div in bs.find_all("button",
                                       {'class': 'js-get-share-url c-article-share-box__button'}): div.decompose()
                for div in bs.find_all("p", {
                    'class': 'js-c-article-share-box__additional-info c-article-share-box__additional-info'}): div.decompose()
                for div in bs.find_all("h2", {'id': 'ethics'}): div.decompose()
                for div in bs.find_all("div", {'class': 'c-article-share-box u-display-none'}): div.decompose()

                content['collect_text'] = bs.text
                if len(bs.find_all("section", {'data-title': 'Abstract'}))!=0:
                    content['abstract'] = bs.find_all("section", {'data-title': 'Abstract'})[0].text
                elif len(bs.find_all("meta", {'name': 'dc.description'}))!=0:
                    content['abstract'] = bs.find_all("meta", {'name': 'dc.description'})[0].text

                else:
                    pdb.set_trace()
                    print ('no abstract')
                json.dump(content, fw)
                fw.write('\n')
print(str(cnt))

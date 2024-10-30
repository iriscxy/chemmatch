import pdb
import time

import requests as rq
import json

header = {"Accept": "application/json", "X-ELS-APIKey": " ",
          "X-ELS-Insttoken": ""}

fww=open('cursor.txt','w')
def fffff(cursor,cnt):
    fw=open('scopus/result/{}.txt'.format(cnt),'w')
    # Break down the variables for the search query
    api_url = "https://api.elsevier.com/content/search/scopus?query=title-abs-key(catalyst)&cursor={}&start=0&view=COMPLETE&count=25".format(cursor)
    response = rq.get(url=api_url, headers=header)
    page = json.loads(response.content.decode("utf-8"))
    for entry in page['search-results']['entry']:
        json.dump(entry,fw)
        fw.write('\n')
    return page['search-results']['cursor']['@next']

cnt=0
cursor='*'
while cursor!='':
    cnt+=1
    print (str(cnt))
    cursor=fffff(cursor,cnt)
    fww.write(cursor+'\n')


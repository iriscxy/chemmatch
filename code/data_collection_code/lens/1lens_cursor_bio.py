import json
import pdb

import requests
import time

url = 'https://api.lens.org/scholarly/search'

request_body = '''{
     "query": {
        "bool": {
            "must": [
            {"match": {"source.asjc_subjects": "Biochemistry"}}
        ],
            "filter": [
                {
                    "term": {
                        "has_abstract": true
                    }
                },
                {
                    "range": {
                        "year_published": [{"gt": "2017"}]
                    }
                }
            ]
        }
    },
    "sort": [
    {
      "year_published": "desc"
    }
  ],
     "size": 500,
     "scroll":"1m"
}'''

headers = {'Authorization': '', 'Content-Type': 'application/json'}


# Recursive function to scroll through paginated results
def scroll(scroll_id, global_cnt):
    print(str(global_cnt))
    fw = open('crawl_bio1/crawl_mate{}.txt'.format(global_cnt), 'w')

    # Change the request_body to prepare for next scroll api call
    # Make sure to append the include fields to make faster response
    if scroll_id is not None:
        global request_body
        request_body = '''{"scroll_id": "%s"}''' % (scroll_id)

    # make api request

    response = requests.post(url, data=request_body, headers=headers)
    # If rate-limited, wait for n seconds and proceed the same scroll id
    # Since scroll time is 1 minutes, it will give sufficient time to wait and proceed
    if response.status_code == requests.codes.too_many_requests:
        time.sleep(8)
        scroll(scroll_id, global_cnt)
    # If the response is not ok here, better to stop here and debug it
    elif response.status_code != requests.codes.ok:
        print(response.json())
    # If the response is ok, do something with the response, take the new scroll id and iterate
    else:
        content = response.json()
        total = content['total']
        if content.get('results') is not None and content['results'] > 0:
            scroll_id = content['scroll_id']  # Extract the new scroll id from response
            # print(json['data'])  # DO something with your data
            for each_case in content['data']:
                json.dump(each_case, fw)
                fw.write('\n')
            global_cnt += 1
            return scroll_id


# start recursive scrolling
scroll_id = None
for global_cnt in range(0, 5000):
    scroll_id = scroll(scroll_id=scroll_id, global_cnt=global_cnt)

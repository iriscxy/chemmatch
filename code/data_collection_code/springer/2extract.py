import fileinput
import json
import os
import pdb
import re
from bs4 import BeautifulSoup

cnt = 0
publisher_set = set()
already_set = set()

fw = open('final.json', 'w')


def clean(abstract, hastitle=False):
    if hastitle:
        title = abstract.split('\t')[0]
        abstract = abstract.split('\t')[1]
        abstract = abstract.strip()
        abstract = abstract.replace('\n', ' ')
        pattern = r'\<.*?\>'
        abstract = re.sub(pattern, '', abstract)
        abstract = re.sub(' +', ' ', abstract)
        abstract = re.sub('\t+', ' ', abstract)
        abstract = title + '\t' + abstract

    else:
        abstract = abstract.strip()
        if abstract.lower().startswith('abstract'):
            abstract = abstract[abstract.lower().index('abstract') + 8:]
        abstract = abstract.replace('\n', ' ')
        pattern = r'\<.*?\>'
        abstract = re.sub(pattern, '', abstract)
        abstract = re.sub(' +', ' ', abstract)
        abstract = re.sub('\t+', ' ', abstract)
    return abstract


def extract_abstract(text):
    match = re.search('(?i)abstract(.*?)\s{2,}', text)
    if match:
        return match.group(0).strip()
    else:
        return ''


def remove_after_spaces(text, num_spaces=5):
    parts = text.split(' ' * num_spaces)
    if len(parts) > 1:
        return parts[0]
    return text


def remove_after_last_period(text):
    last_period_index = text.rfind('.')
    if last_period_index == -1:  # no period found
        return text
    else:
        return text[:last_period_index + 1]


def check_characters(s: str) -> bool:
    # 找到问号的位置
    question_mark_index = s.find("?")

    # 从问号左边开始向左找，找到第一个字母
    left_char = None
    for i in range(question_mark_index - 1, -1, -1):
        if s[i].isalpha():
            left_char = s[i]
            break

    # 从问号右边开始向右找，找到第一个字母
    right_char = None
    for i in range(question_mark_index + 1, len(s)):
        if s[i].isalpha():
            right_char = s[i]
            break

    # 检查找到的字符是否为C或H
    return (left_char in ['C', 'H'] if left_char else False) and (right_char in ['C', 'H'] if right_char else False)


with open('output_springer.json') as file2:
    for line in file2:
        content = json.loads(line)
        doi = content['DOI']
        journal_name = content['container-title']
        abstract = content['abstract']
        try:
            subject = content['subject']
        except:
            subject = []
        if len(abstract.split()) > 20:
            new_content = dict()
            new_content['title'] = content['title'][0]
            new_content['abstract'] = clean(abstract)
            new_content['time'] = content['created']['date-parts'][0]
            new_content['DOI'] = doi
            content['journal_name'] = journal_name
            new_content['subject'] = subject

            json.dump(new_content, fw, ensure_ascii=False)
            fw.write('\n')
            cnt += 1
            print(str(cnt))

with open('crawl_springer.json') as file2:
    for line in file2:
        content = json.loads(line)
        doi = content['DOI']
        text = content['collect_text']
        journal_name = content['container-title']
        try:
            subject = content['subject']
        except:
            subject = []

        abstract = extract_abstract(text)
        if len(abstract.split()) > 20:
            abstract = remove_after_spaces(abstract)
            abstract = remove_after_last_period(abstract)

            new_content = dict()
            new_content['title'] = content['title'][0]
            new_content['abstract'] = clean(abstract)
            new_content['time'] = content['created']['date-parts'][0]
            new_content['DOI'] = doi
            new_content['journal_name'] = journal_name
            new_content['subject'] = subject

            json.dump(new_content, fw, ensure_ascii=False)
            fw.write('\n')
            cnt += 1
            print(str(cnt))

print(str(cnt))

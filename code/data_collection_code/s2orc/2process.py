import json
import glob
import logging
import os
import sys

from tqdm import tqdm

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir)
from utils import clean

import pdb

logging.basicConfig(level=logging.INFO)

abstracts_pattern = "abstracts-part*.jsonl"
papers_pattern = "papers-part*.jsonl"
s2orc_pattern = "s2orc-part*.jsonl"

summ_file = "summ.json"
article_file = "article.json"

abstracts_cache_file = "abstracts_cache.json"
s2orc_cache_file = "s2orc_cache.json"

with open(summ_file, 'w') as sf, open(article_file, 'w') as af:
    pass


def read_jsonl_lines(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            yield json.loads(line)


def extract_fields(paper, abstract, s2orc):
    new_content = {
        'DOI': paper.get('externalids', {}).get('DOI', None) or (
            s2orc.get('externalids', {}).get('doi', None) if s2orc else None),
        'subject': [field['category'] for field in paper.get('s2fieldsofstudy', [])],
        'time': paper.get('publicationdate', None),
        'journal_name': paper.get('journal', None),
        'abstract': abstract.get('abstract', None) if abstract else None,
        'title': paper.get('title', None)
    }
    return new_content


def is_chemistry(fields_of_study):
    if not fields_of_study:
        return False
    return any(field['category'].lower() == 'chemistry' for field in fields_of_study)


def build_corpusid_to_filepath_map(pattern, cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)

    corpusid_to_filepath = {}
    for filepath in glob.glob(pattern):
        with open(filepath, 'r') as f:
            for line_number, record in enumerate(f):
                record = json.loads(record)
                corpusid_to_filepath[str(record['corpusid'])] = (filepath, line_number)

    with open(cache_file, 'w') as f:
        json.dump(corpusid_to_filepath, f)

    return corpusid_to_filepath


print('Building corpusid to file path maps...')
abstracts_map = build_corpusid_to_filepath_map(abstracts_pattern, abstracts_cache_file)
s2orc_map = build_corpusid_to_filepath_map(s2orc_pattern, s2orc_cache_file)

print('Processing papers files...')
papers_files = glob.glob(papers_pattern)
batch_size = 1
fw1 = open('select_abstract.json', 'w')
fw2 = open('select_s2orc.json', 'w')
fw3 = open('select_papers.json', 'w')
for i in range(0, len(papers_files), batch_size):
    print(i)
    relevant_corpusids = set()

    for papers_filepath in papers_files[i:i + batch_size]:
        print(papers_filepath)
        for paper in read_jsonl_lines(papers_filepath):
            if is_chemistry(paper.get('s2fieldsofstudy', [])):
                corpusid = str(paper['corpusid'])
                relevant_corpusids.add(corpusid)
                json.dump(paper,fw3)
                fw3.write('\n')

    abstracts_files_to_read = {abstracts_map[corpusid][0] for corpusid in relevant_corpusids if
                               corpusid in abstracts_map}
    s2orc_files_to_read = {s2orc_map[corpusid][0] for corpusid in relevant_corpusids if corpusid in s2orc_map}

    for abstracts_filepath in abstracts_files_to_read:
        if abstracts_filepath:
            line_set = {abstracts_map[corpusid][1] for corpusid in relevant_corpusids if corpusid in abstracts_map}
            pdb.set_trace()
            with open(abstracts_filepath, 'r') as f:
                for line_number, line in enumerate(f):
                    if line_number in line_set:
                        fw1.write(line)

    for s2orc_filepath in s2orc_files_to_read:
        if s2orc_filepath:
            line_set = {s2orc_map[corpusid][1] for corpusid in relevant_corpusids if corpusid in s2orc_map}
            with open(s2orc_filepath, 'r') as f:
                for line_number, line in enumerate(f):
                    if line_number in line_set:
                        fw2.write(line)

print('Processing completed.')

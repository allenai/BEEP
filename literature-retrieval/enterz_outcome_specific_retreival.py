import time

import pickle
import csv
import math
import datetime
import os
import argparse

EMAILID = "matanhol@gmail.com"
TOOLNAME = ""

from Bio import Entrez
Entrez.email = EMAILID

# Function to retrieve articles from a specified database using a provided query string
# Query string can be a single word/phrase or a list of words/phrase separated using '_'
# Note that if a list of words/phrases is provided, this search will require every term
# to be present in any articles it retrieves (i.e., 'AND' operation for multiple-term lists)
# TODO: Please add your tool name and email ID in the base_url variable
def db_extract(db, query):

    query_enterz = "+".join([x+"[MeSH Terms]" for x in query.split("_")])
    handle = Entrez.esearch(db=db, term=query_enterz, retmax=100000)
    record = Entrez.read(handle)
    num_papers = int(record["Count"])
    id_list = [str(x) for x in record["IdList"]]

    articles_ids = set()
    print(f"{num_papers} papers found.")

    if len(id_list) == num_papers:
        articles_ids.update(id_list)

    else:
        print("cannot get them at once, taking them year by year") # in pubmed when they have > 9999 papers
        today = datetime.datetime.today()
        max_date = today
        min_date = max_date.replace(year=max_date.year-1)
        while len(articles_ids) < num_papers:
            min_date_str = str(min_date.date()).replace(",", "/")
            max_date_str = str(max_date.date()).replace(",", "/")
            handle = Entrez.esearch(db=db, term=query_enterz, retmax=100000,
                                    mindate=min_date_str, maxdate=max_date_str)
            record = Entrez.read(handle)
            id_list = [str(x) for x in record["IdList"]]
            assert len(id_list) == int(record["Count"]), f"failed to get all {min_date} - {max_date} papers"
            articles_ids.update(id_list)

            max_date = min_date
            min_date = max_date.replace(year=max_date.year-1)

    return articles_ids



# Procedure to combine articles retrieved from both PMC and PubMed databases
# To do this combination, PMC article IDs need to be mapped to their corresponding PubMed IDs first
# to avoid double-counting of articles included in both databases
def combine_ids(pmc, pubmed, pmc_ids_map):
    reader = csv.reader(open(pmc_ids_map))
    id_dict = {}
    next(reader, None)
    for row in reader:
        id_dict[row[-4][3:]] = row[-3]
    correct_pmc = set()
    for id in pmc:
        if id not in id_dict or id_dict[id] == '':
            correct_pmc.add('PMC'+id)
            continue
        correct_pmc.add(id_dict[id])
    final_ids = correct_pmc.union(pubmed)
    return final_ids

# Split abstracts according to database they are retrieved from
# This needs to be done to ensure that we are checking the correct database while retrieving text
def split_ids(ids_strings):
    pubmed = []
    pmc = []
    for id_string in ids_strings:
        if id_string.startswith('PMC'):
            pmc.append(id_string[3:])  # Drop PMC prefix since it is no longer needed to distinguish between PubMed/PMC
        else:
            pubmed.append(id_string)
    return pubmed, pmc


def get_abstract_text(subrec):
    if "Abstract" in subrec:
        abstract_text_lines = subrec["Abstract"]["AbstractText"]
        return "\n".join(subrec["Abstract"]["AbstractText"]) + "\n"
    else:
        return ""


def get_abstract_dict(abstract_record):
    pmid = str(abstract_record["MedlineCitation"]["PMID"])
    text = get_abstract_text(abstract_record["MedlineCitation"]["Article"])
    year = int(abstract_record["MedlineCitation"]["DateCompleted"]["Year"])
    return pmid, {"text": text, "year": year}



def retrieve_all_abstracts(id_list, database, outpath, error_log):
    max_query_size = 200  # PubMed only accepts 200 IDs at a time when retrieving abstract text
    print('Retrieval will require {} queries'.format(math.ceil(len(id_list)/float(max_query_size))))
    texts = {}

    total_texts = 0
    for i in range(0, len(id_list), max_query_size):

        start = i
        end = min(len(id_list), start+max_query_size)
        cur_ids = id_list[start:end]

        handle = Entrez.efetch(database, id=cur_ids, retmod="xml")
        record = Entrez.read(handle)


        d = map(get_abstract_dict, record["PubmedArticle"])
        cur_texts = dict((x, y) for x, y in d if y["text"]!="")
        total_texts += len(cur_texts)

        texts.update(cur_texts)

        if end % 1000 == 0 or end == len(id_list):
            print(f'After {end} calls, have {total_texts} abstracts ({end-total_texts} were empty)')
            pickle.dump(texts, open(outpath, 'wb'))
            x=0

    return outpath


def extract_ids(outcome_name, queries, ids_outpath, dbs, pmc_ids_map):
    know_to_handle = set(['pubmed', "pmc"])
    assert set(dbs).issubset(know_to_handle), f"not provided how to handle dbs {set(dbs) - know_to_handle}"

    dbs_ids = {}
    for db in dbs:
        print(db)
        db_ids = set()
        for query in queries:
            print(f"query: {query}")
            db_query_ids = db_extract(db, query)
            db_ids.update(db_query_ids)
        print(f"union of {db}_ids: {len(db_ids)} ids\n")
        dbs_ids[db] = db_ids

    if "pmc" in dbs:
        pubmed_ids = dbs_ids.get("pubmed", set())
        articles_ids = combine_ids(dbs_ids["pmc"], pubmed_ids, pmc_ids_map)
    else:
        articles_ids = dbs_ids["pubmed"]

    print("Final collection for {} has {} articles".format(outcome_name, len(articles_ids)))

    pickle.dump(articles_ids, open(ids_outpath, 'wb'))


def extract_outcome_papers(outcome_name, queries, dbs, out_dir):
    abstracts_outpath = os.path.join(out_dir, f"{outcome_name}_texts_and_dates.pkl")
    ids_outpath = os.path.join(out_dir, f'{outcome_name}_ids.pkl')

    if not os.path.isfile(ids_outpath):
        extract_ids(outcome_name, queries, ids_outpath, dbs)

    articles_ids = pickle.load(open(ids_outpath, "rb"))
    print(f"have {len(articles_ids)} ids")

    # Running text retrieval for IDs retrieved by outcome-specific queries
    pubmed_ids, pmc_ids = split_ids(articles_ids)
    pubmed_ids = sorted(pubmed_ids)
    pmc_ids = sorted(pmc_ids)
    print('{} abstracts will be scraped from PubMed'.format(len(pubmed_ids)))
    print('{} abstracts will be scraped from PMC'.format(len(pmc_ids)))

    error_log = open(f'retrieval_errors.txt', 'w')
    retrieve_all_abstracts(pubmed_ids, 'pubmed', abstracts_outpath, error_log)
    error_log.close()

    return os.path.abspath(abstracts_outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outcome_name', type=str, action='store', required=True, help='name of outcome')
    parser.add_argument('--queries', type=str, action='store', required=True,
                        help='queries for enterz. encapsulated with " ", separated with ,')
    parser.add_argument('--dbs', type=str, nargs="*", action='store', required=True, help='dbs to look in',
                        default=["pmc", "pubmed"])
    parser.add_argument('--out_dir', type=str, action='store', help="directory to save the abstracts")
    parser.add_argument('--PMC_ids_map', type=str, action="store", default="../data/PMC_id_map.csv")
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["queries"] = args_dict["queries"].split(",")
    abstracts_outpath = extract_outcome_papers(**args_dict)
    print(f"abstracts written to {abstracts_outpath}")

    # example:
    # outcome_name = "mortality"
    # queries = "hospital mortality, mortality_risk factors_humans"
    #           --> ["hospital mortality", "mortality_risk factors_humans"]
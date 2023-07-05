import os.path
import pickle
import pip
# Initialize MeSH entity linker to link filtered mentions
import spacy
from scispacy.linking import EntityLinker
import glob
en_core_sci_md_url = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz"
try:
    import en_core_sci_md
except:
    print('downloading "en_core_sci_md"')
    pip.main(["install", en_core_sci_md_url])
    import en_core_sci_md

def link_mentions(filtered_mentions_path, outpath):
    print(f"reading mentions: {filtered_mentions_path}...")
    filtered_mentions = pickle.load(open(filtered_mentions_path, 'rb'))
    print("loading linker...")
    nlp = en_core_sci_md.load(disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    nlp.add_pipe("merge_noun_chunks")
    nlp.add_pipe("scispacy_linker", config={"linker_name": "mesh", "resolve_abbreviations": True})
    #linker = nlp.get_pipe("scispacy_linker")

    # Perform MeSH linking on filtered entities
    for i, file in enumerate(list(filtered_mentions.keys())):
        print('Linking entities for file {} ({})'.format(file, i))
        for sent in filtered_mentions[file]:
            cur_ments = filtered_mentions[file][sent]
            for mention in cur_ments:
                doc = nlp(mention['mention'])
                if not doc.ents:
                    continue
                entities = doc.ents[0]
                cuis = [cui for cui in entities._.kb_ents] #if cui[1]>=0.75]
                if not cuis:
                    continue
                mention['mesh_ids'] = cuis

    pickle.dump(filtered_mentions, open(outpath, 'wb'))

outcome = "mortality"
filtered_mentions_files = glob.glob("./mortality_literature_filtered_mentions*.pkl")
for filtered_file in filtered_mentions_files:
    out_path = filtered_file.replace("filtered", "linked")
    if os.path.isfile(out_path):
        continue
    link_mentions(filtered_file, out_path)
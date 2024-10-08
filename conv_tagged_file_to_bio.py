from app.travel_resolver.libs.nlp.data_processing import from_tagged_file_to_bio_file


INPUT_FILE = "./data/scripting_lcs_1/sentences_10k.txt"
OUTPUT_FILE = "./data/bio/fr.bio/fr.sentences.bio"

tag_entities_pairs = [("<Dep>", "LOC-DEP"), ("<Arr>", "LOC-ARR")]

from_tagged_file_to_bio_file(INPUT_FILE, OUTPUT_FILE, tag_entities_pairs)

from app.travel_resolver.libs.nlp.data_processing import from_tagged_file_to_bio_file


INPUT_FILE = "./data/french_text/1k_unlabeled_samples.txt"
OUTPUT_FILE = "./data/bio/fr.bio/1k_unlabeled_samples.bio"

tag_entities_pairs = [("<Dep>", "LOC-DEP"), ("<Arr>", "LOC-ARR")]

from_tagged_file_to_bio_file(INPUT_FILE, OUTPUT_FILE, tag_entities_pairs)

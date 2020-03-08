#
#
#

import os
import glob
import json
import re
from nltk.tokenize import sent_tokenize


unwanted_pattern = re.compile(r"""([\t\n\\«])""")
section_pattern = re.compile(r"""(Section[:]*[A-Za-zAÄÖÜäü ]*)""")


def split_at_sections(text):
    return re.split(section_pattern, text)


def remove_section_hints(text):
    return re.sub(section_pattern, "", text)


def build_file_from_dir(datadir: str, output_dir: str, remove_sections: bool=True) -> None:
    # Sentencepiece requires one file to compute the BPE from. We need to do this
    # anyways, since the download format is json Aggregated file will be written
    # next to the partial data
    with open(f"{output_dir}sentences.txt", "a") as txtfile:
        for filepath in glob.glob(f"{datadir}*/*/*"):
            if os.path.isdir(filepath):
                continue
            print(f"Reading *** {filepath} ***")
            lines = [line for line in open(filepath, "r")]
            for line in lines:
                text = json.loads(line)["text"]
                for sent in sent_tokenize(text, language='german'):
                    sent = remove_section_hints(sent) if remove_sections else sent
                    sent = [s for s in sent if sent]
                    txtfile.write(sent + "\n")

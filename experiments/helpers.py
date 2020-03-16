#
#
#

import os
import re
import glob
import json
import shutil
import logging
from typing import (
    Dict, List, Tuple
)
from nltk.tokenize import sent_tokenize


logger = logging.getLogger(__name__)
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


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False, save_total_limit=2) -> None:
    if not save_total_limit:
        return
    if save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

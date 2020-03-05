#
#
#

import re
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification, AlbertForMaskedLM
from nltk.tokenize import sent_tokenize


def do_setup(model_type: str = 'albert-base-v2', pretrained_path: str = '../pretrained'):
    if not os.path.exists(pretrained_path):
        os.mkdir(pretrained_path)
        # Load tokenizer and model from pretrained model/voculary
        tokenizer = AlbertTokenizer.from_pretrained(model_type)
        model = TFAlbertForSequenceClassification.from_pretrained(model_type)
        # Save tokenizer and model to pretrained path
        tokenizer.save_pretrained(save_directory=pretrained_path)
        model.save_pretrained(save_directory=pretrained_path)


def load_wiki_data(data_path: str = '../wikidata/extracted/AA/wiki_01', split_sections: bool = False):
    # Assert we have data availbale at data_path
    assert Path(data_path).glob('*/*'), "There are no files available at data_path"

    documents = []
    for line in open(Path(data_path), 'r').readlines():
        if line is not None and not split_sections:
            documents.append([json.loads(line)["text"]])
        else:
            documents.append(json.loads(line)["text"].split("Section::::"))
    return documents


def apply_sequence_length(encoded: dict, max_length: int = 512, cls_id: int = 3):
    assert all([key in encoded.keys() for key in ['input_ids', 'token_type_ids', 'attention_mask']])
    assert max_length % 2 == 0, "Please pass a max_length dividable by 2"

    def find_inner_cls_pos(input_ids, cls_id):
        # Find the second cls_id occurence in input_ids
        # This position is the break point at which we want
        # to measure sentence order prediction task
        inner_cls_pos = 0
        occurence = 0
        for i, input_id in enumerate(input_ids[0]):
            if not input_id == cls_id:
                continue
            else:
                if occurence+1 != 1:
                    occurence += 1
                else:
                    inner_cls_pos = i
                    break
        return inner_cls_pos

    intersection = find_inner_cls_pos(encoded['input_ids'], cls_id)
    # print("Insection found at tensor position", intersection)
    for key in encoded.keys():
        lower_bound = int(intersection-(max_length/2)-1)
        upper_bound = int(intersection+(max_length/2)-1)
        encoded[key] = encoded[key][:, lower_bound:upper_bound]

        assert encoded[key].shape[-1] <= 512
    return encoded


def perform_sentence_order(model, tokenizer, seq_A: str, seq_B: str):
    # Assert we have valid strings and our pretrained models exist
    assert all([isinstance(seq_A, str), isinstance(seq_B, str)]), "Seq_A  and Seq_B has to be of type str"
    assert all([len(seq_A) > 0, len(seq_A) > 0]), "Seq_A  and Seq_B has to be of type str"

    encoded = tokenizer.encode_plus(seq_A, text_pair=seq_B, return_tensors='tf')
    encoded = apply_sequence_length(encoded=encoded, max_length=512)

    # Call the model
    seq_relationship_logits = model(encoded)

    # Since we logits are returned before softmax, we need to, so we can return probabilities
    probs = tf.nn.softmax(seq_relationship_logits)
    return probs


def concatenate_section_sents(sentences: list, section_indicator="Section::::"):
    new_sentences = []
    was_section = False
    for i, sent in enumerate(sentences):
        if section_indicator in sent:
            was_section = True
            if i < len(sentences)-1:
                new_sentences.append(sent+sentences[i+1])
        elif was_section:
            was_section = False
            continue
        else:
            was_section = False
            new_sentences.append(sent)
    return new_sentences


if __name__ == "__main__":
    documents = load_wiki_data(split_sections=False)
    seq_A = documents[0][0]
    print(len(sent_tokenize(text=seq_A)))
    seq_A += documents[7][0]

    sentences = sent_tokenize(text=seq_A)
    sentences = concatenate_section_sents(sentences)

    pretrained_path = '../pretrained'
    assert all([Path(pretrained_path).glob('**/spiece.model'), Path(pretrained_path).glob('**/tf_model.h5')])

    # Load tokenizer and model from pretrained path
    tokenizer = AlbertTokenizer.from_pretrained(pretrained_path)
    model = TFAlbertForSequenceClassification.from_pretrained(pretrained_path)

    all_probs = []
    was_section = []
    for i in range(len(sentences)-1):
        seq_A = sentences[i].replace("\n", "")
        seq_B = sentences[i+1].replace("\n", "")

        print("Section break detected in A", "Section::::" in seq_A, seq_A)
        print("Section break detected in B", "Section::::" in seq_B, seq_B)
        was_section.append("Section::::" in seq_B)
        # seq_A = sent_tokenize(seq_A)[-1] if "Section::::" in seq_A else seq_A
        # seq_B = sent_tokenize(seq_B)[-1] if "Section::::" in seq_B else seq_B
        seq_pattern = re.compile(r"(Section[:]*\w*.)")
        seq_A = re.sub(seq_pattern, '', seq_A).strip()
        seq_B = re.sub(seq_pattern, '', seq_B).strip()
        print("Sequence passed to model: ", seq_A)
        print("Sequence passed to model: ", seq_B)
        # seq_A = seq_A.replace("Section::::", "")
        # seq_B = seq_B.replace("Section::::", "")

        print(i, len(seq_A), len(seq_B))
        probs = perform_sentence_order(model, tokenizer, seq_A=seq_A, seq_B=seq_B)
        print(probs)
        all_probs.append(probs[0, 0, :].numpy())

    results = pd.DataFrame(all_probs)
    results["was_section"] = was_section
    print(results)
    all_probs = np.array(all_probs)
    # print(np.argmax(all_probs[1:, 0])+1)

    # TODO: Perform pairwise sequential comparison and compute local argmin for sequential topic changes
    # TODO: Sequential comparison based on block sizes: n tuple of sentences

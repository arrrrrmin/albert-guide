#
#
#

import os
import logging
import argparse
from transformers import (
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
}

MODEL_VARIANTS = {
    "albert-base-v1": "albert",
    "albert-base-v2": "albert",
    "albert-large-v1": "albert",
    "albert-large-v2": "albert",
    "bert-base-uncased": "bert",
    "bert-large-uncased": "bert",
    "bert-base-cased": "bert",
    "bert-large-cased": "bert"
}


def do_setup(model_type: str = 'albert-base-v2', pretrained_path: str = '../pretrained'):
    # Build the directory when it's not there already
    if not os.path.exists(pretrained_path):
        os.mkdir(pretrained_path)

    # Finds the objects required for the model type (convention from huggingface/transformers)
    config, model, tokenizer = MODEL_CLASSES[MODEL_VARIANTS[model_type]]

    # Load config, model and tokenizer "from_pretrained" and save it to "pretrained_path"
    for object in [config, model, tokenizer]:
        object.from_pretrained(model_type)\
            .save_pretrained(save_directory=pretrained_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_dir", default=None, type=str, required=True,
        help="The input data dir. Should contain the .jsonl files for MMIMDB.",
    )
    parser.add_argument(
        "--model_type", default=None, type=str, required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_VARIANTS.keys()),
    )

    args = parser.parse_args()
    do_setup(model_type=args.model_type, pretrained_path=f"{args.pretrained_dir}/{args.model_type}")

    logging.info("Done")

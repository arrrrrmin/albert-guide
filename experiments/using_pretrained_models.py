#
#

import os
import tensorflow as tf
# import tensorflow_datasets
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification


def do_setup(model_type: str='albert-base-v2', pretrained_path: str='../pretrained'):
    if not os.path.exists(pretrained_path):
        os.mkdir(pretrained_path)
        # Load tokenizer and model from pretrained model/vocabulary
        tokenizer = AlbertTokenizer.from_pretrained(model_type)
        model = TFAlbertForSequenceClassification.from_pretrained(model_type)
        # Save tokenizer and model to pretrained path
        tokenizer.save_pretrained(save_directory=pretrained_path)
        model.save_pretrained(save_directory=pretrained_path)


def load_wiki_data():
    pass


def sentence_experiment(pretrained_path: str='../pretrained'):
    # Load tokenizer and model from pretrained path
    tokenizer = AlbertTokenizer.from_pretrained(pretrained_path)
    model = TFAlbertForSequenceClassification.from_pretrained(pretrained_path)
    seq_A = "Hello, my dog is cute"
    seq_B = "But he's a bit to loud"
    seq_C = "The house will not be be beautiful"

    encodedAB = tokenizer.encode_plus(seq_A, text_pair=seq_B, return_tensors='tf')
    encodedBC = tokenizer.encode_plus(seq_B, text_pair=seq_C, return_tensors='tf')

    seq_relationship_logits_AB = model(encodedAB)
    probsAB = tf.nn.softmax(seq_relationship_logits_AB)
    print(probsAB)

    seq_relationship_logits_BC = model(encodedBC)
    probsBC = tf.nn.softmax(seq_relationship_logits_BC)
    print(probsBC)


if __name__ == "__main__":
    sentence_experiment()

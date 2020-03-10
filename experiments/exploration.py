# A general exploration script, using huggingface/transformers
# More details on transfomers at https://github.com/huggingface/transformers
#


import argparse
import logging
import matplotlib.pyplot as plt

import torch
from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
    AlbertTokenizer
)


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForMaskedLM, AlbertTokenizer),
}


def load_from_file_or_base(object, path):
    if path:
        object = object.from_pretrained(path)
    else:
        logger.info("Filepath for object: %s, not found at %s", object, path)
        raise BaseException("Filepath for object not found")
    return object


def plot_reduced_space(reduced_space, names):

    fig, ax = plt.subplots()
    ax.scatter(reduced_space[:, 0], reduced_space[:, 1])

    for i, txt in enumerate(names):
        ax.annotate(txt, (reduced_space[i, 0], reduced_space[i, 1]))
    plt.show()


def main(model_name_or_path, model_type):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

    config = load_from_file_or_base(config_class, model_name_or_path)
    model = load_from_file_or_base(model_class, model_name_or_path)
    tokenizer = load_from_file_or_base(tokenizer_class, model_name_or_path)

    embeddings = model.albert.embeddings
    print(embeddings)

    wordembeddings = model.albert.get_input_embeddings()
    print(wordembeddings)

    vocab_word2id = tokenizer.get_vocab()
    word_lookup = {id: key for key, id in vocab_word2id.items()}

    n_samples = 1000
    sampled_ids = torch.randint(
        high=len(vocab_word2id.values()), size=(1, n_samples), dtype=torch.long).flatten()
    print("Sampled ids shape", sampled_ids.shape)
    embedded_vectors = wordembeddings(sampled_ids)

    from sklearn.manifold import TSNE
    embedded_reduced = TSNE(n_components=2).fit_transform(embedded_vectors.detach().numpy())
    print(embedded_reduced)

    names = []
    for sid in sampled_ids:
        names.append(word_lookup[int(sid.int())])
    plot_reduced_space(embedded_reduced, names=names)




if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    # required
    arg_parser.add_argument("--model_name_or_path", type=str, required=True)
    arg_parser.add_argument("--model_type", type=str, required=True)

    # other parameters
    arg_parser.add_argument("--model_embedding_visual", action="store_true", required=False)
    args = arg_parser.parse_args()

    main(args.model_name_or_path, model_type=args.model_type)

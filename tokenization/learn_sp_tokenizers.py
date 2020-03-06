# A short demonstration of how to learn a sentencepiece tokenizer
# For more details on sentencepiece tokenization see: https://github.com/google/sentencepiece

import os
import argparse
import sentencepiece
from experiments.helpers import build_file_from_dir


def train_new_tokenizer(
        input_file_path: str, output_model_path: str, vocab_size: int, control_symbols: list
) -> None:
    # Actually learning the model and write it + vocab.txt to output_model_path
    # Suffixes are usually defaulted by the SentencePieceTrainer to <>.model and
    # <>.vocab
    if not os.path.isfile(input_file_path):
        raise BaseException(
            f"Could not train sp tokenizer. Text file is missing. You passed: *** {input_file_path} ***")

    control_symbols = ["[CLS]", "[SEP]", "[MASK]"] if control_symbols is None else control_symbols

    train_command = f"--input={input_file_path} " \
                    f"--model_prefix={output_model_path}spiece " \
                    f"--vocab_size={vocab_size - len(control_symbols)} " \
                    f"--pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 " \
                    f"--user_defined_symbols=(,),”,-,.,–,£,€ " \
                    f"--control_symbols={','.join(control_symbols)} " \
                    f"--shuffle_input_sentence=true --input_sentence_size=10000000 " \
                    f"--character_coverage=0.99995 --model_type=unigram "

    sentencepiece.SentencePieceTrainer.Train(train_command)
    assert (os.path.isfile(f"{output_model_path}spiece.model"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--datadir", type=str, required=True)
    arg_parser.add_argument("--controlsymbols", nargs="*", type=str) # default="[CLS] [SEP] [MASK]"
    args = arg_parser.parse_args()

    # Skip sentences.txt when already exists.
    if not os.path.isfile(f"{args.datadir}/sentences.txt"):
        # In production this process should be done in parallel using ProcessPooler
        build_file_from_dir(datadir=args.datadir, output_dir=args.datadir)

    # Finally learn the actual tokenizer model
    train_new_tokenizer(
        input_file_path="../wikidata/sentences.txt",
        output_model_path="../wikidata/",
        vocab_size=25000,
        control_symbols=args.controlsymbols
    )

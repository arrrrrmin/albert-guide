# albert-guide

A guide to pretrain a new own albert model from scretch

# Pretaining ALBERT models from scretch

> A detailed guide for to get started with ALBERT models as they where intended by google-research.
> Hints for usages in prod can be found at the [end]() of this guide.

# Stages

* [__Environments__, __setups__ and __configurations__](#environments-setups-and-configurations)
    * [Environments](#environments)
    * [Setups](#setups)
    * [Configuration objects](#configuration-objects)
* [__Tokenizers__, __Raws__, __model tasks__ and __records__](#tokenizers-raws-model-tasks-and-records)
    * [Tokenizers](#tokenizers)
    * [Raws](#raws)
    * [Model tasks](#model-tasks)
    * [Records](#records)
* [Main entry __run_pretraining__](#main-entry-run-pretraining)
* [Usage Albert with HF __Transformers__](#usage-albert-with-hf-transformers)

# Environments setups and configuration objects

## Environments

Everything the environment needs to offer is documented in requirements.txt. Here an example where `==X.Y.Z` refers
to the satisfied version of this dependency package.

    transformers
    tensorflow==1.15.2
    tensorflow-gpu==1.15.2
    tensorflow-estimator==1.15.1

The `transformers` package for example will automatically look for the newest version available.
Whereas `tensorflow==1.15.2` will install this exact version, and the therein documented dependencies.
Future note: Packages like [Poetry](https://github.com/python-poetry/poetry) can handle these dependencies
pretty well, as the requirements are growing.

Theres a difference between a local environment and production usage. On a server you most likely don't want to use
an environment, since the server does not need to handly many projects. Thus one can skip the environment and directly
install packages on the system.

For local development it's highly recommanded to use a local environment. When handling different software projects
every environment can define it's own dependencies. For setting those up see [Setups](#Setups)

## Setups

    # Set the virtual environment (please call venv as module with -m)
    python3 -m venv env

    # Enter the environment
    source env/bin/activate

    # Install a pip version and upgrade it (again -m is important)
    python3 -m pip install --upgrade pip

    # Install all packages mentioned in requirements.txt
    # This call should be used with freezed requirements (==X.Y.Z)
    pip3 install -r requirements.txt

    # Upgrade what's possible
    # Execute with --upgrade if you want to have the newest libraries
    # Not recommended if for example tensorflow would upgrade to 2.Y.Z from 1.Y.Z
    # pip3 install -r requirements.txt --upgrade


## Configuration objects

`ALBERT` has a large architecture configuration and also defines a lot of other parameters.
Parameters that suggest how to perform the pretraining. Like `sequence_length`, `masked_lm_prob`,
`dupe_factor` or even newer parameter that didn't exist in original [BERT](https://github.com/google-research/bert)
like `ngram`, `random_next_sentence`, or `poly_power`. `albert_config` is common model architecture
json config.

    "albert_config": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "num_hidden_groups": 1,
        "net_structure_type": 0,
        "gap_size": 0,
        "num_memory_blocks": 0,
        "inner_group_num": 1,
        "down_scale_factor": 1,
        "type_vocab_size": 2,
        "vocab_size": 30000
    }
Additional parameters can be the following:

|Parameter                  |Default    |
|---                        |---        |
|do_lower_case              | true      |
|max_predictions_per_seq    | 20        |
|random_seed                | 12345     |
|dupe_factor                | 2         |
|masked_lm_prob             | 0.15      |
|short_seq_prob             | 0.2       |
|do_permutation             | false     |
|random_next_sentence       | false     |
|do_whole_word_mask         | true      |
|favor_shorter_ngram        | true      |
|ngram                      | 3         |
|optimizer                  | lamb      |
|poly_power                 | 1.0       |
|learning_rate              | 0.00176   |
|max_seq_length             | 512       |
|num_train_steps            | 125000    |
|num_warmup_steps           | 3125      |
|save_checkpoints_steps     | 5000      |
|keep_checkpoint_max        | 5         |

There are even more but these (i think) are the most important for ALBERT. I suggest to also keep theses in a json or
yaml file. If those are kept in a json one can easily read them and build pipelines around the commands provided in the
ALBERT repository.

# Tokenizers raws model tasks and records

## Tokenizers

`ALBERT` supports [`SentencePiece`-Tokenizer](https://github.com/google/sentencepiece/blob/master/python/README.md)
natively. It's fully integrated in the preprocessing pipeline. But to use it, one has to learn a tokenizer, on the
provided data. Google Standard Tokenizers mostly do not support  german and even if they do it's a mulitlingual version
where each Language just is provided with around 1000 individual tokens.

For most NLP applications and corpora a `vocab_size` inbetween `20000` to `40000` should be fine.
The tokenizer itself is trained via:

`python
import os
import logging
import sentencepiece

text_filepath = "path/to/corpus.txt"
model_filepath = "path/to/model/"
vocab_size = 25000
control_symbols = ["[CLS]", "[SEP]", "[MASK]"]

if not os.path.isfile(text_filepath):
    raise BaseException(f"Could not train sp tokenizer, due to missing text file at {text_filepath}")

train_command = f"--input={text_filepath} " \
                f"--model_prefix={model_filepath} " \
                f"--vocab_size={vocab_size - len(control_symbols)} " \
                f"--pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 " \
                f"--user_defined_symbols=(,),”,-,.,–,£,€ " \
                f"--control_symbols={','.join(control_symbols)} " \
                f"--shuffle_input_sentence=true --input_sentence_size=10000000 " \
                f"--character_coverage=0.99995 --model_type=unigram "

logging.info(f"Learning SentencePiece tokenizer with following train command: {train_command}")
sentencepiece.SentencePieceTrainer.Train(train_command)
assert (os.path.isfile(f"{model_filepath}.model"))
`

It'll write two files to `--model_prefix`: `tokenizer.model` and the `tokenizer.vocab`. The vocabulary
has all subtokens and the model is a binary file, to load the model from.

But to train the tokenizer we need a file to pass to `text_filepath`. This can be done with

## Raws

The only thing we need to train a tokenizer is a file that contains all our data. Since the
[`SentencePiece`-Tokenizer](https://github.com/google/sentencepiece/blob/master/python/README.md) is trained on
sentences to detect subtokens in a text, we need to find all sentences that are provided in our data. At this point
we can already think about the way in which we should provide data to tensorflow and the preprocessing pipeline of
`ALBERT`.

In fact there only a very light difference between what the
[`sentencepiece.SentencePieceTrainer.Train`](https://github.com/google/sentencepiece/blob/master/python/sentencepiece.py)
and [`create_pretrain_data.py`](https://github.com/google-research/ALBERT/blob/master/create_pretraining_data.py)
by `ALBERT` original google-research repository. ALBERTs preprocessing pipeline expects the data to be one sentence
per line, just like sentencepiece, but the documents must be seperated by an additional line break (`\n`).

Since `SentencePiece` is fine with no tokens in a line we can format the data such that we only need one file,
instead of two seperate sentences from eachother by `\n` and documents with `\n\n`.

But we still don't know what a sentence is. Classic NLP problem, we need to find what defines a sentence. This question
seem far to complex to tackle at this point, since we just want to format data for the first step on the way to train an
ALBERT model.

Before diving deep into designing regexes for every many many special cases and exceptions in your data: My
recommendation is to pick up `NLTK` as another dependency in your project and add download the tokenizer pickle
from their repository, usoing the `ǹltk.download()` function in your terminal. There are a few languages and it's
easy to handle.

Once your models perform reasonabily with not as many training steps (like `100000` to `150000`) you can tackle
the problem and find your sentences with more accurate ways, that fit your needs.

Now we have our raw.txt file which most likely will be a large file around 1 to 2GB or even larger

## Model tasks

Before we can enter the way in which we address the creation of our preprocessed data, we need to have a look at
what the pretraining tasks are for the model. So let's have a short look at what ALBERT is actually trying to learn,
when we pretrain it.

### Masked LM Prediction

First of all no matter what task we are on there is a new interesting set of parameters in BERT/ALBERT. Since these
models operate on sentences (or `sequences`), we need to set a maximum size, a sequence can have. This parameter
is limited to `512` and is usually either `64`, `128` or `265` if not. This parameter later on
influences the `batch_size`, which determines how large a single batch is that is computed in out turn.

Parameters like `short_seq_prob` are interesting indipendent from what task is
performed. The `short_seq_prob`-Parameter describes at which probability a sequence is shortend down to the length
that is described in `target_seq_length`.

But now let get to the first task: __Masked LM Prediction__ is a task that takes `sentence` as input. Additionally
some other parameters like `do_lower_case` (used in the tokenization), `max_predictions_per_seq`,
`do_whole_word_mask` and `masked_lm_prob` are passed, to fine configure this task. This task also exists in the
original BERT model and aims to MASK tokens within a sentence. The model then tries to predict the words from the
known (passed words).

Here is an example that comes from the original [BERT repository](https://github.com/google-research/BERT):

`
Input: the man went to the [MASK1] . he bought a [MASK2] of milk.
Labels: [MASK1] = store; [MASK2] = gallon
`

In reality there is no token named `[MASK1]` or `[MASK2]`. These will be masked with the same token called
`[MASK]`. In binary elements per token this would look like `[0,0,0,0,0,1,0,0,0,0,1,0,0,0]`. All tokens at
positions maked with `1` should be predicted, whereas tokens marked with `0` are passed as ids to the model.

Additionally the parameter `masked_lm_prob` tells how many of a sequences available tokens are masked. This is done
before padding the sequence up to 512, or what ever is set as `max_seq_length`. So `masked_lm_prob` is applied
to the length of the raw sequence, not the padded length.

Another interessting parameter is `do_whole_word_mask`. This tells the pretraining data process to only mask full
words, instead of subwords. Tokenizers like `sentencepiece` are using special characters to separate subtokens from
each other and also mark a subtoken needs some other token combined to be understood as a full token/word. In
`sentencepiece` this special character is `▁` (looks like a normal underscore but it is not). This character
marks a subword, so when `do_whole_word_mask` is used this token is used to find out if the token before or after
should be masked too. Like this it's possible to mask full words instead of subwords.


### Sentence Order Prediction

Sentence Order Prediction (SOP) is a new task in ALBERT and didn't exist in BERT original. It replaces the 
Next Sentence Prediction (NSP) task. Basically both tasks aim to learn relationships between segments (sentences). Since 
Masked LM Prediction (MLM) does only care about tokens within a certain segement, these tasks are designed to learn 
information about language properties that are formed from sequences of tokens. On could say it's an inter sequencially
designed task. 
Whereas BERT originally tried to predict wether *"two segments appear consecutively in the original text"*. 
[Yang et. al.](https://arxiv.org/abs/1906.08237) & [Liu et. al.](https://arxiv.org/abs/1907.11692) eliminated that task
and observed an impovements on all finetune tasks. [Lan et. al.](https://arxiv.org/pdf/1909.11942) showed that the 
reason for this behaviour is coming from the fact that the task is to easy. The observation basically showed that NSP
benefited from MLM as this single sequence task was already learning a good portion of topical information. Which
helped when predicting the similarity between two sentences. NSP simply learned to use the already existing knowledge 
of topics. Maybe this could have also beent tackled with a different sampling strategy but anyways, they replaced it
with SOP. 

This task does not predict wether a a segment is the next, wether two segments are in the correct order. Negative 
samples are generated by swapping two consecutive sentences. Positive samples are taken from two a document as it is.
SOP performes far more stable in contrast to NSP.

As in the original BERT model sentences are marked by `[CLS]` for the start of the first segment, `[SEP]` 
for the end of this segment, and another `[SEP]` token for the end of the second segment. The process works like
this:

* Choose two sentences from the corpus
    * When `random_next_sentence` is set we'll want to use a random sentence from a another document
    * When `random_next_sentence` is not set we'll just offset by one and take the one after the correct sentence 
* Apply subword tokenization
    * `▁` helps to find out wether to aggregate tokens when `whole_word_masking` is set
* Now finalize segments with `[CLS]` ... `[SEP]` ... `[SEP]`
* Wrap all of this in a training `instance` and add `next_sentence_labels` with either `0` or `1`
    * `0` labels the second segment as consecutively
    * `1` labels the second segment as incorrect/*random*
    * Note that `next_sentence_labels` was moved from BERT unchanged
        * i guess to make it easier for libs like Huggingface a or Spacy to update their code


## Records

Now that we fully understand what the model should do, we can create `instances`, that will be written to a special
format used in `tensoflow` to pass data, called `tfrecords`. In such a record each training instance looks like
this:

    INFO:tensorflow:tokens: [CLS] a man went to a [MASK] [SEP] he bou ▁ght a [MASK] of milk [SEP]
    INFO:tensorflow:input_ids: 2 13 48 1082 2090 18275 7893 13 4 3 37 325 328 3235 48 4 44 1131 3 0 0 0 0 ...
    INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 ...
    INFO:tensorflow:segment_ids: 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 ...
    INFO:tensorflow:token_boundary: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ...
    INFO:tensorflow:masked_lm_positions: 8 13 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
    INFO:tensorflow:masked_lm_ids: 65 2636 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 
    INFO:tensorflow:masked_lm_weights: 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
    INFO:tensorflow:next_sentence_labels: 1

`Instances` hold data for MLM and SOP. And are written to `tfrecord`-files. In this case 
`next_sentence_labels` is refering to *Sentence Order Prediction*.


# Main entry run pretraining

When we are having everything in place `Configu`, `Tokenizer` and `Pretraining Data`, we can start pretraining a 
fresh model. The shell call is using a provided script 
[`run_pretraining.py`](https://github.com/google-research/ALBERT/blob/master/run_pretraining.py)

The call involves some parameters, for example `input_file`, which can also be a directory, where multiple `records`
are created by [`data_preperation.py`](https://github.com/google-research/ALBERT/blob/master/create_pretraining_data.py)
. `output_dir` is the directory, which will contain out model. In case we have started training and aborted somehow,
we can use `init_checkpoint` to continue. Keep an eye on `save_checkpoints_steps`, since it tells us how frequent the
model is saved, during training. `num_warmup_steps` can be set to 2.5% of `num_train_steps`. This is the number of steps
the model will apply a lower learning rate, until it reaches the passed `learning_rate` parameter.

    pip install -r albert/requirements.txt
    python -m albert.run_pretraining \
        --input_file=... \
        --output_dir=... \
        --init_checkpoint=... \
        --albert_config_file=... \
        --do_train \
        --do_eval \
        --train_batch_size=4096 \
        --eval_batch_size=64 \
        --max_seq_length=512 \
        --max_predictions_per_seq=20 \
        --optimizer='lamb' \
        --learning_rate=.00176 \
        --num_train_steps=125000 \
        --num_warmup_steps=3125 \
        --save_checkpoints_steps=5000


# Usage Albert with HF Transformers

In order to use Albert as efficiently as possible I'd recommend to use 
[Hugging Face (HF) Transformers](https://github.com/huggingface/transformers). It's an open source library, that 
provides many very useful interfaces and functionalities, that make our live easier as NLP developers/researchers.
The guys at huggingface are very up to date on what's going on and also provide useful advice in case something is 
unclear. A very nice community.












# Learning to Write

The official repo for [Learning to Write](https://arxiv.org/pdf/1805.06087.pdf), published in ACL, 2018.

You can view samples at our [demo site](https://ari-holtzman.github.io/l2w-demo/).

If you use this in your own work, please [cite us](https://ari-holtzman.github.io/l2w-demo/holtzman2018l2w.bib).

## Requirements

```bash
# (0) Setup fresh python3 environment using your method of choice.

# (1) Install pytorch 0.4 using instructions from pytorch.org

# (2) Install torchtext at specific commit
cd ../
git clone https://github.com/pytorch/text.git
cd text/
git reset --hard 36310207f5ca45c87e3192ace320353816ead618
cd ../l2w/
pip3 install ../text/
```

## Generating Pre-Trained Models

You can download pre-trained models and sample data [here](https://drive.google.com/open?id=1ehvfjP4VPWo9UZotbSeYVWEq3yAoe_76) and [here](https://drive.google.com/open?id=10oS_0vazJspY0nAg7tMqzVlcNzRsTB9d). Unzip them, and put them in the root of the repo. 

You can then generate by running:

```bash
# TorontoBooks
python generate.py --data data/tbooks_sample.txt --lm models/tbooks/lm.pt --dic models/tbooks/vocab.pickle --print  --cuda --scorers models/tbooks/best_scorer_weights.tsv

# TripAdvisor
python generate.py --data data/trip_sample.txt --lm models/trip/lm.pt --dic models/trip/vocab.pickle --print  --cuda --scorers models/trip/best_scorer_weights.tsv
```

## Training Your Own

### Split Data

Split data into lm-train, disc-train, valid, and test

```bash
python scripts/split_data.py /path/to/data.txt /path/to/dataset/directory/
```

### Build a Shared Dictionary

The base language model and all discriminators use the same vocabulary, so we have to build it ahead of time.

```bash
python utils/make_dic.py /path/to/training_set.txt path/to/save/vocab.pickle --max_vocab 100000
```

### Train Language Model

We have to train the base generator first, because two of the discriminators rely on generations from the LM for their training data.

```bash
python adaptive_softmax/train.py --cuda --data /path/to/data --dic /path/to/dictionary --cutoffs 4000 40000 --tied --nlayers 2
```

### Train Discriminators

#### Data 

First, let's build all the required data files for training all the discriminators.

For the main data, you need to run this script and generate from the language model.

```bash
# (1) Run the main processing script. Options for different kinds of datasets viewable using --help
python scripts/make_cc_version.py /path/to/data/

# (2) Run the script that generates data from the LM
bash scripts/gen_lm_data.sh /path/to/data/ /path/to/lm.pt /path/to/vocab.pickle
```

For the entailment data, first concatenate all the '.txt' version of all the [SNLI](https://nlp.stanford.edu/projects/snli/) and [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) data (including train, dev, and test), but watchout to not include the column headers. Then

```bash
# (1) Format the data 
python scripts/create_nli_dataset.py /path/to/concatenated/data.txt /path/to/nli_output.tsv

# (2) Split the data
python scripts/split_data.py /path/to/nli_output.tsv /path/to/nli_data/ --no_disc_train --valid_frac 0.1 --test_frac 0.1  
```

#### Repetition 

```bash
# (1) Make rep data
python scripts/create_classifier_dataset.py /path/to/disc_data/ /path/to/save/rep_data/ --comp lm

# (2) Train model
python trainers/train_classifier.py /path/to/rep_data/ --save_to /path/to/save/model.pt --dic /path/to/vocab.pickle  --fix_embeddings --adam --ranking_loss --train_prefixes
```

#### Entailment

The entailment data was  already generated in the "Data" section, so now we can just train the model.

```bash
python trainers/train_entailment_classifier.py /path/to/nli_data/ --save_to /path/to/save/model.pt --dic /path/to/vocab.pickle --adagrad --batch_size 4 --lr 0.01 --num_epochs 100
```

#### Relevance

```bash
# (1) Make rel data
python scripts/create_classifier_dataset.py /path/to/disc_data/ /path/to/save/rel_data/ --comp random

# (2) Train model
python trainers/train_classifier.py /path/to/rel_data/ --save_to /path/to/save/model.pt --dic /path/to/vocab.pickle \
--decider_type cnncontext --adam  --ranking_loss --train_prefixes
```

#### Lexical Style

The lexical style module uses the exact same data as the repetition module, but doesn't view data as sequences of cosine similarities. Thus, we can train it on the data we made for the repetition classifier:

```bash
# (1) Train model
python trainers/train_classifier.py /path/to/rep_data/ --save_to /path/to/save/model.pt --dic /path/to/vocab.pickle \
--decider_type poolending --adam --ranking_loss --train_prefixes
```

### Train Discriminator Weightings

First you have to make a weights file, in the following (tab separated) format:

```
1	SCORER_PATH	SCORER_CLASS	/path/to/model.pt
```

SCORER_PATH and SCORER_CLASS are word_rep.context_scorer & ContextScorer respectively for all modules, except the entailment module. For the entailment module SCORER_PATH and SCORER_CLASS are entailment.entail_scorer_new & EntailmentScorer.

For an example, see our [pre-trained models](https://drive.google.com/open?id=1ehvfjP4VPWo9UZotbSeYVWEq3yAoe_76).

Once you have a scorer_weights.tsv simply run:

```bash
python scripts/create_classifier_dataset.py /path/to/disc_data/ /path/to/save/weight_data/ --comp none

python generate.py --cuda  --data /path/to/weight_data/valid.tsv --lm /path/to/lm.pt --dic /path/to/vocab.pickle  --scorers /path/to/scorer_weights.tsv --print --learn
```

### Generate

```bash
python generate.py --cuda --data /path/to/weight_data/test.tsv --lm /path/to/lm.pt --dic /path/to/vocab.pickle  --scorers /path/to/scorer_weights.tsv --print
```

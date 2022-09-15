# Word Embedding Similarity Calculator
This repository contains a command-line Python script (`get_similarities.py`) that allows the user to calculate the similarity of two words (or sequences of words) based on their word embeddings.  This includes the similarity between two words, or the similarity between a word and its context. The current version of the tool is compatible with [GloVe](https://nlp.stanford.edu/projects/glove/) and [fastText](https://fasttext.cc/) embeddings.

## Cosine Similarity
The standard metric used for calculating the similarity between two embeddings is the cosine similarity $\text{Sim}_{\text{cos}}$ between their vectors $\bm{a}$ and $\bm{b}$, given by the following equation:

$$\text{Sim}_{\text{cos}} = \frac{\bm{a}\cdot\bm{b}}{|\bm{a}||\bm{b}|} $$

Note that this is equivalent to $\cos(\theta_{a,b})$, the cosine of the angle $\theta_{a,b}$ between the two vectors $\bm{a}$ and $\bm{b}$.

If one or both of the sequences for which the cosine similarity is to be calculated is made up of more than one word, following convention, the mean of the embeddings is used. For words not appearing in the embeddings, there are several options:

1. Not calculating the cosine similarity for this input.
2. For words containing punctuation, automatically trying to split the word into sub-word tokens.
3. Ignoring these words and calculating the mean embedding using the other words in each sequence&mdash;this only works if there is at least one word that appears in the embeddings. Note that this can also be used in combination with (2), in which case sub-words not present in the embeddings are ignored.

### Sub-word tokens
The current approach to splitting the word into sub-word tokens is the following:

1. Try to use the full word.
2. If the full word is not present, split the word before any punctuation. For example, `Alice's` would be split into `Alice` and `'s`, `didn't` would be split into `didn` and `'t`, `sub-word` would be split into `sub` and `-word`. This often works for contractions and possessives.
3. If not all of these sub-words are present, split into the pre-punctuation, punctuation, and post-punctuation parts of the word.  For example, `Alice's` would be split into `Alice`, `'`, and `s`; `didn't` would be split into `didn`, `'`, and `t`, `sub-word` would be split into `sub`, `-`, and `word`.

## How to use

### Step 1: Download embeddings
This tool is designed to use [GloVe](https://nlp.stanford.edu/projects/glove/) and [fastText](https://fasttext.cc/) embeddings, which can be downloaded directly from their respective websites. With the [fastText vectors for 157 languages](https://fasttext.cc/docs/en/crawl-vectors.html), make sure to download the `text` rather than `bin` files&mdash;once extracted from the zip file, these will have a `.vec` extension.

Note: if your embeddings are uncased, you should use the `--uncased` argument when running the script.

### Step 2: Prepare your input file(s)
This tool can be used to calculate the cosine similarity between two words, or between a word (or sequence of words) and its context. The principle is the same for both. 

The format of the input files must follow a strict form. Firstly, the input file must be a plain text file. Each input should be provided on a separate line.

For calculating the similarity between two words, the line should start with one word, and the second word should be encapsulated by asterisks (`*`). For example, to calculate the similarity between `easy` and `difficult`, the relevant line in the input file would take one of the following forms:

```
easy *difficult*
```

```
difficult *easy*
```

For calculating the cosine similarity between a word or a sequence of words and its context, the word or sequence should be encapsulated by asterisks (`*`). See the following examples:

```
Sometimes running experiments is easy and sometimes it is *difficult*
```

```
Sometimes running experiments is easy and sometimes *it is difficult*
```

```
Sometimes running experiments is *easy* and sometimes it is difficult
```

```
Sometimes *running experiments is easy and sometimes it is difficult*
```

The default is to only use the context preceding the sequence encapsulated by asterisks, but the following context can also be included using the `--following_context` or `-f` argument (see below for full list of arguments).

### Step 3: Run the script
The script is run from the command line. For example, to use a set of embeddings `latest_embeddings.txt` to calculate the cosine similarity for an input file `input_file.txt` and output to an `output` directory, the following would be entered into the command line:

```
python get_similarities.py -i input.txt -o output -m latest_embeddings.txt
```

This will output a file with the name `input_file.latest_embeddings` to the output directory. The output file is a tab-separated values file, with three columns: 

* `FullSentence`: The input as written in the input file with asterisks removed.
* `TargetWords`: The part of the input encapsulated by asterisks.
* `CosineSimilarity`: The calculated cosine similarity.

The arguments for the script are explained below.

### Required arguments
* `--stimuli` (`-i`) or `--stimuli_list` (`-ii`): The input(s). Note that a `--stimuli_list` argument will override a `--stimuli` argument.
    * `--stimuli` (`-i`): The path to a file containing the inputs for which cosine similarity is to be calculated (in the format described above).
    * `--stimuli_list` (`-ii`): The path to a file listing the paths to one or more input files, separated by line.
* `--output_directory` (`-o`): The output directory.
* `--embeddings` (`-m`) or `--embeddings_list` (`-mm`): The embedding(s) to use to calculate cosine similarity.  Note that a `--model_list` argument will override a `--embeddings` argument.
    * `--model` (`-m`):  The path to a file containing the embeddings to be used to calculate cosine similarity (as mentioned above, currently, GloVe and fastText embeddings are supported).
    * `--embeddings_list` (`-mm`): The path to a file listing the paths to the embeddings to be used, separated by line.

### Optional Arguments
* `--following_context` (`-f`): Whether or not to include the following context&mdash;that is, anything following the asterisk-encapsulated sequence&mdash;when calculating cosine similarity. The default is `False`; including this argument will switch this to `True`.
* `--try_subwords_in_context` (`-cs`): If a word in the context (i.e. outside the asterisks) is not present in the embeddings, try to decompose it into its sub-word tokens (as described above). The default is `False`; including this argument will switch this to `True`.
* `--ignore_oov_in_context` (`-cv`): If a word in the context (i.e. outside the asterisks) is not present in the embeddings, ignore it and calculate cosine similarity anyway (note: this can be used in combination with `-cs`, as described above). The default is `False`; including this argument will switch this to `True`.
* `--try_subwords_in_target` (`-ts`): If a word in the target (i.e. encapsulated by asterisks) is not present in the embeddings, try to decompose it into its sub-word tokens (as described above). The default is `False`; including this argument will switch this to `True`.
* `--ignore_oov_in_target` (`-tv`): If a word in the target (i.e. encapsulated by asterisks) is not present in the embeddings, ignore it and calculate cosine similarity anyway (note: this can be used in combination with `-cs`, as described above). The default is `False`; including this argument will switch this to `True`.
* `--uncased`: Use if downloaded embeddings are uncased. This will convert all inputs and embeddings to lower-case (for writing systems with this feature). Note that this will be applied to all embeddings when the `--embeddings_list`/`-mm` argument is used, so if multiple embeddings are being used, cased and uncased embeddings should be run separately.


## Requirements
The Word Embedding Similarity Calculator was written for `Python 3.8` and requires the `scipy` package.

## How to cite

```
@misc{michaelov_2022_WordEmbeddingSimilarity,
  title={Word Embedding Similarity Calculator},
  author={Michaelov, James A.},
  year={2022}
}
```

## References
* Pennington, J., Socher, R., & Manning, C. D. (2014). [GloVe: Global vectors for word representation](https://aclanthology.org/D14-1162/). In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).
* Mikolov, T., Grave, É., Bojanowski, P., Puhrsch, C., & Joulin, A. (2018). [Advances in Pre-Training Distributed Word Representations](https://aclanthology.org/L18-1008/). In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.
* Grave, É., Bojanowski, P., Gupta, P., Joulin, A., & Mikolov, T. (2018). [Learning Word Vectors for 157 Languages](https://aclanthology.org/L18-1550/). In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.

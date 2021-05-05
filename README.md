# mbr-nmt
Minimum Bayes-Risk Decoding for Neural Machine Translation

# Installation
```
git clone git@github.com:Roxot/mbr-nmt.git
cd mbr-nmt
pip install .
```

# Basic usage
```
mbr-nmt translate -s sample_filename -n num_samples -u utility -o output_filename [-c candidates_file]
```

Because sentences are not translated in order, the output of `mbr-nmt translate` should be ordered correctly first. This can be done using `mbr-nmt convert` as follows:

```
mbr-nmt convert -f mbr-nmt -i mbr_nmt_translate_output -o translations_file
```

## Example usage
```
mbr-nmt convert -f fairseq -i fairseq-output -o samples-300.en --merge-subwords --detruecase --detokenize
mbr-nmt translate -s samples-300.en -n 300 -u beer -o translations.en.mbr
mbr-nmt convert -f mbr-nmt -i translations.en.mbr -o translations.en 
```

# Samples file
The samples file should contain an equal amount of translation samples per input sentence that you wish to translate. Translation samples should be separated by a newline. Example:
```
samples-3.en:
<input-1-sample-1>
<input-1-sample-2>
<input-1-sample-3>
<input-2-sample-1>
<input-2-sample-2>
<input-2-sample-3>
...
```

We provide support for parsing the output of fairseq when sampling to convert the fairseq output into a format readable by `mbr-nmt translate` using `mbr-nmt convert`. See `mbr-nmt convert --help` for more info. Example:

```
mbr-nmt convert -f fairseq -i fairseq_output_file -o samples_file --merge-subwords --detruecase --detokenize
```

# Candidates file
Optionally, one can provide a separate list of candidates from which translations are to be selected. If not set, unique samples will be used as candidates instead. The candidates file is more flexible than the samples file in that it supports different number of candidates per input sentence. Each set of candidates must be preceded by the line `NC=<number_of_candidates>` and candidates should be separated by a newline. Example:
```
candidates.en
NC=2
<input-1-candidate-1>
<input-1-candidate-2>
NC=3
<input-2-candidate-1>
<input-2-candidate-2>
<input-2-candidate-3>
...
```

# Utilities
See `mbr-nmt translate --help` for a full list of utilities currently supported. Some utilities require to be installed first.

## BEER installation
Follow the installation instructions at https://github.com/stanojevic/beer. Make sure that `$BEER_HOME` is visible to `mbr-nmt translate`

# mbr-nmt
Minimum Bayes-Risk Decoding for Neural Machine Translation


# Utilities

## Unigram precision

## BEER

## METEOR

## SacreBLEU

We support 
* BLEU
* ChrF
* TER
as wrappers around [sacrebleu](https://github.com/mjpost/sacrebleu).

## BLEURT

To use [Google's BLEURT](https://github.com/google-research/bleurt), install it:

```bash
pip install --upgrade pip  # ensures that pip is current
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

Then create the BLEURT object (for which you will need a model checkpoint, they come along with the repository you cloned)

```python
from mbr_nmt.utility import BLEURT
utility = BLEURT('/home/wferrei1/github/bleurt/bleurt/test_checkpoint')
utility('George went to school by bike today .'.split(), 'George went to school by bike today .'.split())
# This should be > 0.9
```

**On Jupyter notebook** If you play with BLEURT as utility from a jupyter notebook you will get some strange error about command line options, see [here](https://github.com/google-research/bleurt/issues/4). It turns out it's resolved if you do `import sys; sys.argv = sys.argv[:1]`. 


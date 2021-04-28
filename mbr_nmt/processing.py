import sacrebleu

def merge_subwords(line, style="fairseq"):
    """
    :param line: translation with tokens split by spaces.
    :param style: style of subwording.
    """
    if style != "fairseq": raise NotImplementedError
    return ''.join(line.split(' ')).replace('▁', ' ').strip()

def v13_tokenizer(line):
    return sacrebleu.tokenize_13a(line)

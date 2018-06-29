"""
What? NLP giving language TO humans? What?
"""

# builtins
import re

# 3rd party
from nltk.tokenize import sent_tokenize


def detokenize(text: str):
    # punctuation hacks
    text = re.sub('\s*,\s*', ', ', text)
    text = re.sub('\s*\.\s*', '. ', text)
    text = re.sub('\s*\?\s*\?\s*', '??', text)
    text = re.sub('\s*\?\s*\!\s*', '?!', text)
    text = re.sub('\s+\?', '?', text)
    text = re.sub('\s+\!', '!', text)
    text = re.sub('\s*\'\s*', '\'', text)
    text = re.sub(r'\([â€™]+\s*\)*', '\'', text)
    text = re.sub('</s>', '', text)
    text = re.sub('<beg>', '', text)
    text = re.sub('<end>', '', text)
    text = re.sub(r' &apos;', "'", text)
    text = re.sub(r'&apos;', "'", text)
    text = re.sub(r'&lt;', "<", text)
    text = re.sub(r'&gt;', ">", text)
    text = re.sub(r'&#91;', "[", text)
    text = re.sub(r'&#93;', "[", text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&amp;', "&", text)
    text = re.sub(r' n\'t', 'n\'t', text)


    # sentence capitalization
    sents = sent_tokenize(text)
    for i, s in enumerate(sents):
        s = s.strip()
        sents[i] = s[0].upper() + s[1:]
    text = ' '.join(sents)

    return text


def main():
    # test string
    text = 'i like , to , dance at La Rumba . don \' t you know ? hello ? ! ? ! ? ?'
    print('orig:', text)
    print('detk:', detokenize(text))


if __name__ == '__main__':
    main()

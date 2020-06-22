"""
Creates a corpus from Wikipedia dump file.
Inspired by:
https://github.com/panyang/Wikipedia_Word2vec/blob/master/v1/process_wiki.py
"""
import sys

from gensim.corpora import WikiCorpus


def make_corpus(in_f, out_f):
    """Convert Wikipedia xml dump file to text corpus"""
    wiki = WikiCorpus(in_f)
    with open(out_f, 'w') as output:
        i = 0
        for text in wiki.get_texts():
            output.write('\n'.join(text) + '\n')
            i += 1
            if not i % 10000:
                print(f'Processed {i} articles', end='\r')
    print('\nProcessing complete!')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python make_wiki_corpus.py <wikipedia_dump_file> <processed_text_file>')
        sys.exit(1)
    in_f, out_f = sys.argv[1:]
    make_corpus(in_f, out_f)


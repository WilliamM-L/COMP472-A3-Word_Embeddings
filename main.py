import os

import gensim.downloader

def main():
    gensim.downloader.BASE_DIR = os.path.dirname(os.path.realpath(__file__)) + '\\gensim-data'
    model = gensim.downloader.load('word2vec-google-news-300')
    print(model.distance('man', 'woman'))

if __name__ == '__main__':
    main()
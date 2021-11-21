import csv
import os
# set env var so it knows where to look for the gensim model data
import random

from enums import Label

gensim_data_path = os.path.dirname(os.path.realpath(__file__)) + '\\gensim-data'
os.environ['GENSIM_DATA_DIR'] = gensim_data_path
# ^ must be done before we import gensim!
import gensim.downloader


def test_single_model(model_name='word2vec-google-news-300'):
    with open(f'logs\\{model_name}-details.csv', 'w') as logger:
        guess_count = 0
        correct_counter = 0
        gensim.downloader.BASE_DIR = gensim_data_path
        model = gensim.downloader.load(model_name)
        with open('datasets\\synonyms.csv') as dataset_file:
            data = csv.DictReader(dataset_file)
            for row in data:
                question = row['question']
                answer = row['answer']
                words_to_compare = []
                answer_index = -1
                for key, word in row.items():
                    # only comparing words whose key is a number
                    if key.isdigit():
                        words_to_compare.append(word)
                        # adding the winning index
                        if word == answer:
                            answer_index = int(key)
                try:
                    similarities = model.distances(question, words_to_compare)
                    print(similarities)
                    lowest_score = 2
                    for score in similarities:
                        print(score)
                except KeyError:
                    # will get here if any words in the list are not part of the model, must check each one manually :(
                    print('Could not find word!')
                    lowest_score = 2
                    model_guess_index = 8
                    for index, word_to_compare in enumerate(words_to_compare):
                        try:
                            score = model.distance(word_to_compare, answer)
                        except KeyError:
                            continue
                        if score < lowest_score:
                            lowest_score = score
                            model_guess_index = index
                    # no matter if we get it right or not, we've been guessing since we don't know all words
                    guess_count += 1

                    if lowest_score == 2:
                        # got a KeyError for every invocation -> answer is unknown or all 4 guess words are unknown
                        model_guess_index = random.randint(0, len(words_to_compare)-1)
                        # the label couldn't be Correct even if the guess is right since we don't even know one guess word or the answer word
                        label = Label.Guess
                    else:
                        if model_guess_index == answer_index:
                            label = Label.Correct
                            correct_counter += 1
            logger.write(f'{question},{answer},{words_to_compare[model_guess_index]},{str(label)}')



def main():
    test_single_model()


if __name__ == '__main__':
    main()

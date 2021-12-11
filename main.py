import csv
import os
# set env var so it knows where to look for the gensim model data
import random

from enums import Label

gensim_data_path = os.path.dirname(os.path.realpath(__file__)) + '\\gensim-data'
os.environ['GENSIM_DATA_DIR'] = gensim_data_path
# ^ must be done before we import gensim!
import gensim.downloader


initial_value = 9999


def test_single_model(model_name='word2vec-google-news-300'):
    correct_counter = 0
    non_guessing_counter = 0
    with open(f'logs\\{model_name}-details.csv', 'w') as logger:
        gensim.downloader.BASE_DIR = gensim_data_path
        model = gensim.downloader.load(model_name)
        with open('datasets\\synonyms.csv') as dataset_file:
            data = csv.DictReader(dataset_file)
            for row in data:
                question = row['question']
                answer = row['answer']
                words_to_compare, answer_index = preprocess_words_to_compare(row, answer)

                try:
                    similarities = model.distances(question, words_to_compare)
                    print(similarities)
                    non_guessing_counter += 1
                    label, correct_counter, model_guess_index = evaluating_similarities_without_guessing(similarities, answer_index, correct_counter)

                except KeyError:
                    # will get here if any words in the list are not part of the model, must check each one manually :(
                    print('Guessing!')
                    label, correct_counter, model_guess_index = evaluating_similarities_with_guessing(model, words_to_compare, answer, answer_index, correct_counter)

                finally:
                    logger.write(f'{question},{answer},{words_to_compare[model_guess_index]},{label}\n')

    with open('logs\\analysis.csv', 'a') as analysis_logger:
        analysis_logger.write(f'{model_name},{len(model)},{correct_counter},{non_guessing_counter},{correct_counter/non_guessing_counter}\n')


def evaluating_similarities_without_guessing(similarities, answer_index, correct_counter):
    lowest_score = initial_value
    model_guess_index = initial_value
    for index, score in enumerate(similarities):
        if score < lowest_score:
            lowest_score = score
            model_guess_index = index

    if model_guess_index == answer_index:
        correct_counter += 1
        label = Label.Correct
    else:
        label = Label.Wrong
    return label, correct_counter, model_guess_index


def evaluating_similarities_with_guessing(model, words_to_compare, answer, answer_index, correct_counter):
    lowest_score = initial_value
    model_guess_index = initial_value
    for index, word_to_compare in enumerate(words_to_compare):
        try:
            score = model.distance(word_to_compare, answer)
        except KeyError:
            continue
        if score < lowest_score:
            lowest_score = score
            model_guess_index = index

    if lowest_score == initial_value:
        # got a KeyError for every invocation -> answer is unknown or all 4 guess words are unknown
        model_guess_index = random.randint(0, len(words_to_compare) - 1)
        # the label couldn't be Correct even if the guess is right since we don't even know one guess word or the answer word
    return Label.Guess, correct_counter, model_guess_index


def preprocess_words_to_compare(row, answer):
    words_to_compare = []
    answer_index = initial_value
    for key, word in row.items():
        # only comparing words whose key is a number
        if key.isdigit():
            words_to_compare.append(word)
            # adding the winning index
            if word == answer:
                answer_index = int(key)
    return words_to_compare, answer_index


def task_2_different_corpus():
    test_single_model(model_name='fasttext-wiki-news-subwords-300')
    test_single_model(model_name='glove-wiki-gigaword-300')


def task_2_same_corpus():
    test_single_model(model_name='glove-twitter-100')
    test_single_model(model_name='glove-twitter-200')


def main():
    test_single_model()
    task_2_different_corpus()
    task_2_same_corpus()


if __name__ == '__main__':
    main()

from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu


def load_doc(filename):
    """
    load doc into memory
    :param filename:
    :return:
    """
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def load_set(filename):
    """
    Load a pre-defined list of photo identifiers
    :param filename:
    :return:
    """
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


def load_clean_descriptions(filename, dataset):
    """
    Load clean descriptions into memory
    :param filename:
    :param dataset:
    :return:
    """
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions


def load_photo_features(filename, dataset):
    """
    Load photo features
    :param filename:
    :param dataset:
    :return:
    """
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features


def to_lines(descriptions):
    """
    Covert a dictionary of clean descriptions to a list of descriptions
    :param descriptions:
    :return:
    """
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


def create_tokenizer(descriptions):
    """
    Fit a tokenizer given caption descriptions
    :param descriptions:
    :return:
    """
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def max_length(descriptions):
    """
    # calculate the length of the description with the most words
    :param descriptions:
    :return:
    """
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def word_for_id(integer, tokenizer):
    """
    # map an integer to a word
    :param integer:
    :param tokenizer:
    :return:
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    """
    # generate a description for an image
    :param model:
    :param tokenizer:
    :param photo:
    :param max_length:
    :return:
    """
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    """
    evaluate the skill of the model
    :param model: Model
    :param descriptions: Descriptions
    :param photos: photos
    :param tokenizer: tokenizer
    :param max_length: max_length
    :return:
    """
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        print(yhat)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def main():
    """
    The main function serves as the starting point for program execution.
    It usually controls program execution by directing the calls to other functions in the program.

    :return:
    """
    # load training dataset (6K)
    filename = 'Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print('Dataset: %d' % len(train))
    # descriptions
    train_descriptions = load_clean_descriptions('descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))
    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    mlength = max_length(train_descriptions)
    print('Description Length: %d' % mlength)

    # prepare test set
    # load test set
    filename = 'Flickr_8k.testImages.txt'
    test = load_set(filename)
    print('Dataset: %d' % len(test))
    # descriptions
    test_descriptions = load_clean_descriptions('descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
    # photo features
    test_features = load_photo_features('features.pkl', test)
    print('Photos: test=%d' % len(test_features))

    # load the model
    filename = 'final_model.h5'
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, test_descriptions, test_features, tokenizer, mlength)


if __name__ == '__main__':
    main()


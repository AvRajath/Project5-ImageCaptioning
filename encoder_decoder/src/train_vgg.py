import tensorflow
import keras
from os import listdir
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from pickle import load, dump
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from numpy import array


def extract_features(directory):
    """
    extract features from each photo in the directory
    :param directory: directory path that has all the images downloaded from Flicker8k dataset.
    :return: a dictionary of features along with their
    """
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print('>%s' % name)
        pass
    return features


def load_doc(filename):
    """
    Load doc into memory
    :param filename: Name of the file containing all tokens
    :return: document
    """
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def load_descriptions(doc):
    """
    Extract descriptions for images
    :param doc: Input document with token of the images
    :return: mapping images with list of tokens
    """
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # remove filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


def clean_descriptions(descriptions):
    """
    Prepare translation table for removing punctuation
    :param descriptions: This cleans all the descriptions by removing punctuations and stop words etc.
    :return: None
    """
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)


def to_vocabulary(descriptions):
    """
    build a list of all description strings.
    :param descriptions:
    :return:
    """
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


def save_descriptions(descriptions, filename):
    """
    save descriptions to file, one per line.
    :param descriptions:
    :param filename:
    :return:
    """
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_set(filename):
    """
    load a pre-defined list of photo identifiers
    :param filename: Load set
    :return: dataset
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
    load clean descriptions into memory
    :param filename: filename with descriptions
    :param dataset: dataset that has descriptions
    :return: cleaned descriptions
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
    load photo features
    :param filename: file that has features
    :param dataset: image-ids
    :return: dictionary of features and image-id
    """
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features


def to_lines(descriptions):
    """
    convert a dictionary of clean descriptions to a list of descriptions
    :param descriptions: descriptions
    :return: None
    """
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


def create_tokenizer(descriptions):
    """
    fit a tokenizer given caption descriptions
    :param descriptions: descriptions
    :return: tokenizer
    """
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    """
    Create sequences of images, input sequences and output words for an image
    :param tokenizer: This is the tokenizer
    :param max_length: Max length
    :param desc_list: List of descriptions
    :param photo: Photo
    :param vocab_size: Size of the vocabulary
    :return: sequences of the words.
    """
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    """
    Data generator, intended to be used in a call to model.fit_generator()
    :param descriptions: descriptions of the data
    :param photos: photos of the data generator
    :param tokenizer: Tokenizer
    :param max_length: This is the max length
    :param vocab_size: Vocab size
    :return: None
    """
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
            yield ([in_img, in_seq], out_word)
            pass


def max_length(descriptions):
    """
    Calculate the length of the description with the most words
    :param descriptions: Descriptions
    :return: maximum length of the description.
    """
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def main():

    """
    The main function serves as the starting point for program execution.
    It usually controls program execution by directing the calls to other functions in the program.

    :return:
    """
    # extract features from all images
    directory = 'Flicker8k_Dataset'
    features = extract_features(directory)
    print('Extracted Features: %d' % len(features))

    # save to file
    dump(features, open('features.pkl', 'wb'))

    filename = 'Flickr8k.token.txt'
    # load descriptions
    doc = load_doc(filename)

    # parse descriptions
    descriptions = load_descriptions(doc)
    print('Loaded: %d ' % len(descriptions))

    # clean descriptions
    clean_descriptions(descriptions)

    # save descriptions
    save_descriptions(descriptions, 'descriptions.txt')

    filename = 'Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print('Dataset: %d' % len(train))

    # descriptions
    train_descriptions = load_clean_descriptions('descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))

    # photo features
    train_features = load_photo_features('features.pkl', train)
    print('Photos: train=%d' % len(train_features))

    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)


if __name__ == '__main__':
    main()
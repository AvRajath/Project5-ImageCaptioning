from train_vgg import *
from pickle import dump, load
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from tensorflow.keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint


def extract_features(directory):
    """
    extract features from each photo in the directory
    :param directory: directory path that has all the images downloaded from Flicker8k dataset.
    :return: a dictionary of features along with their
    """
    # load the model
    model = ResNet50()
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
        # prepare the image for the ResNet model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print('>%s' % name)

    return features


def define_model(vocab_size, max_length):
    """
    Define the captioning model
    :param vocab_size: Vocab-size
    :param max_length: Length of the maximum.
    :return: ResNet Model
    """
    # feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


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
    pass


if __name__ == '__main__':
    main()

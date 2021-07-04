import os
import cv2
import pytesseract
import numpy as np
import re

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import naive_bayes
from sklearn import neighbors
from spellchecker import SpellChecker
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from pytesseract import Output
from PIL import Image

def __run__():
    # Load stopwords list
    stopwords = open("stopwords.txt", 'r').read().split()

    # List of all document convert to stirng
    all_document_text = []

    # List of all document's class : impot/compte/fonciere/paie/attestion
    document_class = []

    # Retrieve all used files
    for path in os.listdir("data_pdf"):
        for file in os.listdir("data_pdf/" + path):
            print("[INFO]   Openning %s file for data preparation" % file)

            # Load pdf images
            images = convert_from_path("data_pdf/" + path + "/" + file, use_pdftocairo=True, strict=False)
            document = ""

            # For each images of each documents
            for image in images:
                image = np.array(image)

                # Image rotation if needed
                image = rotate_image(image)

                # Converting image in gray scale
                image_gs = get_gray_scale(image)

                custom_config = r"--oem 3 --psm 6"

                # Convert image to string
                print("[INFO]      Converting %s images to string" % file)
                result_txt = pytesseract.image_to_string(image_gs, config=custom_config, lang='fra')

                # Clean string retrieved
                print("[INFO]      Cleaning %s text representation" % file)
                result_txt_cleaned = rm_digit_and_spe_char(result_txt, stopwords)

                document += " " + result_txt_cleaned

            all_document_text.append(document) # X
            document_class.append(path) # y

    # Load class and text in numpay array
    y = np.array(document_class)
    X = vectorize_documents(all_document_text)

    # Create a pipeline model instance
    pipeline = {"SVM": svm.SVC(kernel='linear', C=1, random_state=42),
                "Naive Bayes": naive_bayes.GaussianNB(),
                "KNN": neighbors.KNeighborsClassifier(n_neighbors=3)}

    for model in pipeline:
        print("[INFO]   Training for model %s" % model)
        pipeline_ml(pipeline[model], X, y)


def rotate_image(image):
    """
        Analyses and corrects the rotation of the image

        :param image: image on which the rotation will be performed
        :type image: `np.array`
    """

    # Get somes information about the image rotation
    rotation = pytesseract.image_to_osd(image, output_type=Output.DICT)

    # Extract angle rotation
    angle = rotation["rotate"]

    # Pytesseract only return four type of rotation : 0, 90, 180, 270
    # We check the rotate returned and if needed we rotate the image with the correct angle
    if (angle == '0'):
        image = image
    elif (angle == '90'):
        print("Rotating 90°")
        image = Image.rotate(image,90,(255,255,255))
    elif (angle == '180'):
        image = Image.rotate(image,180,(255,255,255))
        print("Rotating 180")
    elif (angle == '270'):
        image = Image.rotate(image,90,(255,255,255))
        print("Rotating 270")

    return image

def pipeline_ml(model, X, y):
    """
        Train a model with cross validation method and print the model result

    :param model: the trainning model
    :param X: Attributes
    :type X: `np.ndarray`
    :param y: Target
    :type y: `np.ndarray`
    """

    #
    # First method : Use cross val scores to evaluate our model (similar to k-fold)
    #
    scores = cross_val_score(model, X, y, cv=2)
    print("[INFO]       Cross validation score result for SVM model : %s" % scores.mean())


    #
    # Second method : Use k-fold to evaluate our model
    #
    mean_predict = []

    # Setup KFold parameters
    kf = KFold(n_splits=6)

    # Train and test the SVM model for each segment created by kf.split(X)
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        model.fit(X_train, y_train)
        mean_predict.append(model.score(X_test, y_test))

    mean = sum(mean_predict) / len(mean_predict)
    print("[INFO]       K-Fold score result for SVM model : %s" % mean)

def vectorize_documents(documents):
    """
        Vectorize documents as bag of words

    :param documents: List of all documents retrieved
    :rtype: list

    :return: Vectorized document
    :rtype: `np.ndarray`
    """

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    return X.toarray()


def get_gray_scale(image):
    """
        Return image to gray scale

    :param image: image to transform
    :return: image transformed to gray scale
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def rm_digit_and_spe_char(text, stopwords):
    """
        Prepare and clean text :
            - Remove digit
            - Remove special character
            - Remove stopword
            - Correct typo with pyspellchecher

    :param text: text to clean
    :param stopwords: list of stopword used to remove stopword in text

    :return: cleaned text
    :rtype: str
    """

    spell = SpellChecker(language="fr", distance=1)  # fix distance to 1 for shorter run times

    text_output = " "
    for word in text.split():
        word = re.sub(r'\d+', "", word) # remove digital char
        word = re.sub(r'[\@!-+°—"-_*()=,;:./?…|<>«»]', " ", word) # remove special character
        word = word.lower() # normalize to lower case
    
        # Check if the word is myspell
        word = spell.correction(word) if spell.unknown([word]) else word
        
        if word not in stopwords:
            text_output += " " + word

    return text_output


if __name__ == '__main__':
    __run__()

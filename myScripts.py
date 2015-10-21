
import csv
import string
import re
import numpy as np
import subprocess

def process_text(text):
    text = re.sub(r"('[^ ]*)|(__EOS__)", "", text) # remove anything that follows an apostrophe, and __EOS__
    text = text.replace(' - ', ' ')
    text = text.replace('-', '_')
    text = re.sub(r'[^a-zA-Z01-9_ \n]', ' ', text) # remove punctuation and other unusual characters
    return text.lower() # convert all words to lower case

def join_files(filenames_in, filename_out):
    output = open(filename_out, 'w')
    for filename in filenames_in:
        with open(filename, 'rb') as csvfile:
            data = csvfile.read()
            text = re.sub(r'[ ]+', ' ', data) # replace all multiple spaces with a single space
            output.write(text)
    output.close() 


def preprocess_interview_text(prefix, filename_in, filename_out):

    # open files to output interview text to
    filenames = [prefix + 'author.txt', prefix + 'movie.txt', prefix + 'music.txt', prefix + 'general.txt']
    author = open(filenames[0], 'w')
    movie = open(filenames[1], 'w')
    music = open(filenames[2], 'w')
    general = open(filenames[3], 'w')

    with open(filename_in, 'rb') as csvfile:
        data = list(csv.reader(csvfile))
        for x in range(1, len(data)): # skip first line
            text = process_text(data[x][1]) # unprocessed text is 2nd feature
            category = int(data[x][2]) # category is 3rd feature
            if category == 0:
                author.write(text)
            elif category == 1:   
                movie.write(text)       
            elif category == 2:   
                music.write(text)      
            else:
                general.write(text)  
    author.close()
    movie.close()
    music.close()
    general.close()
    return filenames


class FeatureStats(object):

    def __init__(self, distances):
        self.max = max(distances)
        self.min = min(distances)
        distance_array = np.array(distances)
        self.median = np.median(distance_array)
        self.mean = np.mean(distance_array)

    def __str__(self):
        return (
            'min = {}\n' 
            'max = {}\n' 
            'median = {}\n'
            'mean = {}\n'
            .format(self.min, self.max, self.median, self.mean)
        )

def compute_distance_from_feature(vector_filename, feature, text):
    """
    Computes the distance using word2vec to the specified feature for 
    each individual word in the provided text. 
    Returns a set of statistics for distance between the text and the feature
    """
    # process text and write to temporary file
    tmp_file = open('tmp_input.txt', 'w+')
    tmp_file.write(process_text(text))
    tmp_file.close()

    # read in output of distance to feature for each word
    try:
        distance_output = subprocess.check_output(['./distance2FileWords', vector_filename, feature, 'tmp_input.txt'])
        distances = []
        for i in distance_output.split():
            if float(i) != -1.0:
                distances.append(float(i))
        return FeatureStats(distances)
    except subprocess.CalledProcessError:
        print 'Unknown error occurred while computing distances'
    return None

def compute_feature_matrix(filename_in, filename_out, features): 
    output = open(filename_out, 'w+')
    with open(filename_in, 'rb') as csvfile:
        data = list(csv.reader(csvfile))
        for x in range(1, 200): # skip first line
            for feature in features:
                stats = compute_distance_from_feature('vValidation.bin', feature, data[x][1])
                output.write('{}, '.format(stats.max))
            category = int(data[x][2]) # category is 3rd feature
            output.write('{}\n'.format(category))
    output.close()

if __name__=='__main__':
    # filenames = preprocess_interview_text('all_training_data/', 'ml_dataset_train.csv', 'training.txt')
    # join_files(filenames, 'all_training_data/text_corpus.txt')
    # stats = compute_distance_from_feature('vValidation.bin', 'music', text)
    # print stats
    compute_feature_matrix('ml_dataset_train.csv', 'train_features.csv', ['music', 'movie', 'novel'])





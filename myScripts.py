
import csv
import string
import re
import numpy as np
import subprocess
import operator
import pickle 

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
        distance_array = np.array(distances)
        distance_array.sort()
        self.max = distance_array[-1]
        self.min = distance_array[0]
        self.mean = np.mean(distance_array)
        self.top5 = np.mean(distance_array[-8:])

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




# The Good Stuff    

def write_processed_text(out_file, dataset_filename):
    with open(dataset_filename, 'rb') as csvfile:
        data = list(csv.reader(csvfile))
        for x in range(1, 10000): # skip first line
            text = process_text(data[x][1]) # unprocessed text is 2nd feature
            word_count = len(text.split())
            out_file.write('{} {}\n'.format(word_count, text))

def prepare_distance_input(dataset_filename, features, out_filename):
    out_file = open(out_filename, 'w+')
    out_file.write('0 {} {}\n'.format(len(features), ' '.join(features)))
    write_processed_text(out_file, dataset_filename)
    out_file.close()

def compute_features(vector_filename, input_filename):
    subprocess.call(['./distance2Words', vector_filename, input_filename])
    print 'Finished calculating distances...'
    

def compute_statistics(num_features, out_filename, classification_file=None):

    if classification_file is not None:
        csvfile = open(classification_file, 'rb');
        classes = list(csv.reader(csvfile))

    output = open(out_filename, 'w+')
    with open('output.txt', 'rb') as datafile:
        i = 0
        while True:
            if i % 1000 == 0:
                print 'trial', i
            line = datafile.readline()
            if line == '':
                break;
            values = line.replace('\n', '').split(',')
            d = [[] for x in xrange(num_features)]
            feature_index = 0; 
            for value in values:
                if value != '':
                    d[feature_index].append(float(value))
                    feature_index = (feature_index + 1) % num_features
               
            # currently uses just max value
            for j in range(num_features):
                if len(d[j]) == 0:
                    print 'empty at {}'.format(i)
                stats = FeatureStats(d[j])
                output.write('{},{},{}'.format(stats.top5, stats.max, stats.mean))
                if j != num_features - 1:
                    output.write(',')

            if classification_file is not None:
                category = int(classes[i + 1][2]) # category is 3rd feature
                output.write(',{}\n'.format(category))
            else:
                output.write('\n')
            i = i + 1


    output.close()
    print 'finished computing statistics'


def get_best_features(num_features):
    words = pickle.load(open('Data/feature_idf_score.txt', 'rb'))
    sortedwords = sorted(words.items(), key=operator.itemgetter(1))
    features = []
    for bestw in sortedwords[-num_features:]:
        splitted = bestw[0].split()
        for w in splitted:
            features.append(str(w))
    return features + ['movie', 'music', 'novel']

def replace_features(file_in, file_out): 

    output = open(file_out, 'w+')
    with open(file_in, 'r') as csvfile:

        while True: 
            line = csvfile.readline().replace('\n', '')
            if line == '':
                break
            data = line.split(',')
            if data[-1] == '2' or data[-1] == '1':
                output.write('{}\n'.format(line))

    output.close()

def extract_features_for_test_set(features): 
    prepare_distance_input('ml_dataset_test_in.csv', features, 'test_input.txt')
    compute_features('vTraining.bin', 'test_input.txt')
    compute_statistics(len(features), 'test_features.csv')

def extract_features_for_train_set(features):
    prepare_distance_input('ml_dataset_train.csv', features, 'train_input.txt')
    compute_features('vTraining.bin', 'train_input.txt')
    compute_statistics(len(features), 'train_features.csv', classification_file='ml_dataset_train.csv')

if __name__=='__main__':
    features = get_best_features(10)
    extract_features_for_train_set(features)


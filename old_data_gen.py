import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, SimpleRNN, LSTM, Embedding, Dropout
from keras import utils as np_utils
import os.path
from os import path

np.random.seed(0)


class DataGenerator:
    def __init__(self):

        self.consonnants = ['B', 'D', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T']
        self.vowels = ['A', 'E', 'I', 'O', 'U']
        self.present_features_indices = ['']
        self.letters = dict()

        self.words_ortho = dict()
        self.words_ortho[''] = []

        self.non_words_ortho = dict()
        self.non_words_ortho[''] = []

        self.semantic_categories = dict()
        self.semantic_categories[''] = []
        self.semantic_units = []

        self.X = []
        self.y = []

        self.generate_orth_rep_for_letters()
        self.generate_words(128)
        # self.generate_words(128, non_word=True)
        self.generate_semantics()
        self.generate_inputs_and_outputs()

    def generate_orth_rep_for_letters(self):
        all_letters = self.consonnants + self.vowels
        for letter in all_letters:
            rep = [0] * 6

            present_features = ''
            while(present_features in self.present_features_indices):
                present_features = "".join(
                    map(str, np.random.randint(0, 6, (2,))))

            self.present_features_indices.append(present_features)
            rep[int(present_features[0])] = 1
            rep[int(present_features[1])] = 1

            self.letters[letter] = rep

    def generate_words(self, num_words, non_word=False):
        def get_word(non_word):
            if (non_word):
                co_i = np.random.randint(len(self.consonnants))
                vo_i = np.random.randint(len(self.vowels), size=2)
                l1 = self.vowels[vo_i[0]]
                l2 = self.consonnants[co_i]
                l3 = self.vowels[vo_i[1]]
                word = l1 + l2 + l3
                ortho = self.letters[l1] + self.letters[l2] + self.letters[l3]
            else:
                co_i = np.random.randint(len(self.consonnants), size=2)
                vo_i = np.random.randint(len(self.vowels))
                l1 = self.consonnants[co_i[0]]
                l2 = self.vowels[vo_i]
                l3 = self.consonnants[co_i[1]]
                word = l1 + l2 + l3
                ortho = self.letters[l1] + self.letters[l2] + self.letters[l3]
            return word, ortho

        words_dict = self.words_ortho if not non_word else self.non_words_ortho

        for i in range(num_words):
            word = ''
            ortho = []
            while(word in words_dict):
                word, ortho = get_word(non_word)

            words_dict[word] = ortho
        print(words_dict)
        del words_dict['']

    def similarity(self, input, others):
        min = 1000
        sum = min
        input = list(map(int, list(input)))
        if input == []:
            return 0
        for other in others:
            other = list(map(int, list(other)))
            if other != []:
                inp3 = np.logical_xor(input, other)
                sum = np.sum(inp3)
            else:
                sum = 1000

            if sum < min:
                min = sum
        return sum

    def generate_semantics(self):
        MAX_NCATEGORIES = 500
        MAX_NMEMBERS = 100
        MAX_NFEATURES = 2000

        nFeatures = 100  # number of features per pattern
        nCategories = 8  # number of clusters (prototypes)
        nMembers = 16  # number of exemplars per cluster
        minProbOn = 0.1  # maximum sparcity of prototype
        maxProbOn = 0.1  # minimum sparcity of prototype
        minDiff = 4  # minimum bit-wise difference among exemplars
        minProbDistort = 0.2  # min prob that feature is regenerated
        maxProbDistort = 0.4  # max prob that feature is regenerated
        sparse = 1  # generate output in "sparse" (unit numbers) format
        minOn = 1  # Min Number of units to be on in the exemplar
        maxOn = 100 	# Max number of units to be on in the exemplar
        maxWCatDiff = 100

        def flip(prob):
            if np.random.uniform(0, 1) < prob:
                return True
            return False

        proto = [0] * MAX_NFEATURES
        item = [0] * MAX_NFEATURES
        cats = [[[0] * MAX_NFEATURES] * MAX_NMEMBERS] * MAX_NCATEGORIES
        for c in range(nCategories):
            # print(c, '--------------------------------')
            probDistort = minProbDistort
            if c < nCategories / 2:
                probOn = minProbOn
            else:
                probOn = maxProbOn

            if c < nCategories / 2:
                probDistort = minProbDistort
            else:
                probDistort = maxProbDistort
            probOn = minProbOn + c * \
                (maxProbOn-minProbOn) / (float((nCategories-1)))

            #  generate new prototype (with exact correct number of ON features)

            # for f in range(nFeatures):
            #   proto.append(0)

            nOn = int(0.5 + probOn * nFeatures)
            n = 0
            while n < nOn:
                f = int(np.random.uniform(0, 1) * nFeatures)
                if(proto[f] == 0):
                    proto[f] = 1
                    n += 1

            m = 0
            attempts = 0
            # while (m < nMembers and attempts < 1000): # add max attemps = 1000
            while (m < nMembers):  # add max attemps = 1000
                attempts += 1
                # generate new potential item
                new = True
                numOn = 0
                for f in range(nFeatures):
                    if flip(probDistort):
                        item[f] = int(flip(probOn))
                    else:
                        item[f] = proto[f]

                    if item[f] == 1:
                        numOn += 1

                if (numOn > maxOn):
                    new = False
                if (numOn < minOn):
                    new = False

                om = 0
                while om < m and new:
                    nDiff = 0
                    for f in range(nFeatures):
                        if item[f] != cats[c][om][f]:
                            nDiff += 1
                    if (nDiff < minDiff):
                        new = False
                    if (nDiff > maxWCatDiff):
                        new = False
                    om += 1

                if not new:
                    print(c, ' Failed diff check 1')
                    continue

                oc = 0
                while oc < c and new:
                    om = 0
                    while om < nMembers and new:
                        nDiff = 0
                        for f in range(nFeatures):
                            if item[f] != cats[oc][om][f]:
                                nDiff += 1
                        if nDiff < minDiff:
                            new = False
                        om += 1
                    oc += 1

                if not new:
                    # print(c, ' Failed diff check 2', ' nDiff ', nDiff < minDiff, new)
                    continue

                for f in range(nFeatures):
                    cats[c][m][f] = item[f]
                m += 1

            if attempts == 1000:
                print('Max attempts reached')

            # print(proto[:nFeatures])
            # print('Members')

            for m in range(nMembers):
                self.semantic_units.append(cats[c][m][:nFeatures])
        self.semantic_units = np.asarray(self.semantic_units)

    def generate_inputs_and_outputs(self):
        # inputs
        word_ortho = list(self.words_ortho.values())
        # non_word_ortho = list(self.non_words_ortho.values())
        # all_ortho = np.asarray(word_ortho + non_word_ortho)
        all_ortho = np.asarray(word_ortho)

        # outputs
        # word_sem = self.semantic_units
        non_word_sem = [[0] * 100] * (len(self.non_words_ortho))

        def convert(elem):
            return list(map(int, list(elem)))

        # all_sem = np.asarray(list(map(convert, word_sem)) + non_word_sem)
        # all_sem = np.asarray(list(map(convert, word_sem)))
        # print(np.asarray(self.semantic_units))
        self.X, self.y = self.shuffle(all_ortho, self.semantic_units)

    def shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]


dg = DataGenerator()
dg.X.shape, dg.y.shape

words = dg.words_ortho
for key in words.keys():
    words[key] = "".join(map(str, words[key]))


def write_to_ex(row):
    if len(row) == 100:  # target
        data_type = "T: "
    elif len(row) == 18:  # input
        data_type = "I: "
    return data_type + " ".join(map(str, row))


list_x = list(map(list, list(dg.X)))

x = list(map(write_to_ex, list_x))
y = list(map(write_to_ex, dg.y))


def write_to_file(x, y):
    out = ''
    for i in range(128):
        orth = "".join(map(str, dg.X[i]))
        name = get_name(orth)
        out += 'name: ' + '{' + str(i) + '_' + name + '}' + \
            '\n' + x[i] + '\n' + y[i] + '\n' + ';\n'

    f = open("c90-dominance.ex", "w")
    f.write(out)
    f.close()


def get_name(inp):
    for key in words.keys():
        if words[key] == inp:
            return key
    raise Error


write_to_file(x, y)

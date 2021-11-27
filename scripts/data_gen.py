import numpy as np
import os
import math


class DataGenerator:
    def __init__(self, seed):
        np.random.seed(seed)
        self.seed = seed
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
        self.generate_words(128, non_word=True)
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

        ortho_str = ''
        orthos = [ortho_str]
        word = ''
        for _ in range(num_words):
            while(word in words_dict.keys() or ortho_str in orthos):
                word, ortho = get_word(non_word)
                ortho_str = "".join(list(map(str, ortho)))

            words_dict[word] = ortho
            orthos.append(ortho_str)

        assert len(words_dict.keys()) == len(np.unique(np.asarray(list(words_dict.keys())))
                                             ), f'{len(words_dict.keys())} != {len(np.unique(np.asarray(list(words_dict.keys()))))}'

        del words_dict['']

        word_ortho = list(self.words_ortho.values())
        orthos = []
        for o in word_ortho:
            o = list(map(str, o))
            o = "".join(o)
            orthos.append(o)
        orthos = np.asarray(orthos)

        assert len(np.unique(orthos)) == len(
            word_ortho), f'{len(np.unique(orthos))} != {len(word_ortho)}'

    def similarity(self, input, others, typ):
        min = 1000
        sum = min
        if typ == 'y':
            inp = list(map(int, list(input['ex'])))
        else:
            inp = list(map(int, list(input)))
        if inp == []:
            return 0
        for other in others:
            if typ == 'y':
                oth = list(map(int, list(other['ex'])))
            else:
                oth = list(map(int, list(other)))
            if oth != []:
                sum = np.sum(abs(np.asarray(inp) - np.asarray(oth)))
            else:
                sum = 1000

            if sum < 4:
                if typ == 'y':
                    print(" ".join(list(map(str, input['ex']))))
                    print(" ".join(list(map(str, other['ex']))))
                    print('------------------------------------')
            if sum == 0:
                if typ != 'y':
                    print(" ".join(list(map(str, input))))
                    print(" ".join(list(map(str, other))))
                    print('------------------------------------')

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

        cats = [[[0] * nFeatures] * nMembers] * nCategories
        for c in range(nCategories):
            proto = [0] * nFeatures
            dominance = ''
            category_num = c
            # print(c, '--------------------------------')
            probDistort = minProbDistort
            if c < nCategories / 2:
                probOn = minProbOn
            else:
                probOn = maxProbOn

            # if c < nCategories / 2:
            #     probDistort = minProbDistort  # high dominance
            #     dominance = 'high_dominance'
            # else:
            #     probDistort = maxProbDistort  # low dominance
            #     dominance = 'low_dominance'

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
            members_info_map = dict()
            # while (m < nMembers and attempts < 1000): # add max attemps = 1000
            while (m < nMembers):  # add max attemps = 1000
                item = [0] * nFeatures
                if m < nMembers / 2:
                    probDistort = minProbDistort  # high dominance
                    dominance = 'dominance-high'
                else:
                    probDistort = maxProbDistort  # low dominance
                    dominance = 'dominance-low'
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
                    # for f in range(nFeatures):
                    nDiff = np.sum(
                        abs(np.asarray(item) - np.asarray(cats[c][om])))
                    # if item[f] != cats[c][om][f]:
                    #     nDiff += 1
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
                        nDiff = np.sum(
                            abs(np.asarray(item) - np.asarray(cats[c][om])))
                        if nDiff < minDiff:
                            new = False
                        om += 1
                    oc += 1

                if not new:
                    # print(c, ' Failed diff check 2', ' nDiff ', nDiff < minDiff, new)
                    continue

                cats[c][m] = item[:nFeatures]
                freq = 1 if m % 2 == 0 else 4
                richness = np.sum(cats[c][m][:nFeatures])
                members_info_map[m] = {'category': 'category-' + str(category_num),
                                       'dominance': dominance, 'ex': cats[c][m][:nFeatures], 'freq': freq, 'richness': richness}
                m += 1

            if attempts == 1000:
                print('Max attempts reached')

            # print(proto[:nFeatures])
            # print('Members')

            for m in range(nMembers):
                semantic_unit = members_info_map[m]
                self.semantic_units.append(semantic_unit)
        # self.semantic_units = np.asarray(self.semantic_units)

    def generate_inputs_and_outputs(self):
        # inputs
        word_ortho = list(self.words_ortho.values())
        non_word_ortho = np.asarray(list(self.non_words_ortho.values()))
        orthos = []
        for o in word_ortho:
            o = list(map(str, o))
            o = "".join(o)
            orthos.append(o)
        orthos = np.asarray(orthos)
        assert len(np.unique(orthos)) == len(word_ortho)
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

        p = np.random.permutation(128)
        # self.X = all_ortho[p]
        self.X = all_ortho
        self.y = self.semantic_units

        idx = 0
        res = ''
        for ortho in self.X:
            sem = self.y[p[idx]]
            list_ortho = list(ortho)
            inp = self.write_to_ex(list_ortho)
            out = self.write_to_ex(sem['ex'])
            word = self.get_name(list_ortho, self.words_ortho)
            name = f"{str(idx)}_{word}_{sem['dominance']}_{str(sem['category'])}_rich-{sem['richness']}_dataSeed-{self.seed}_freq-{str(sem['freq'])}_word"
            res += 'name: ' + '{' + name + '}' + '\nfreq: ' + str(sem['freq']) + \
                '\n' + inp + '\n' + out + '\n' + ';\n'
            idx += 1
        self.word_ex_file_buf = res

        f = open("../data/ex/words" + ".ex", "w")
        f.write(self.word_ex_file_buf)

        idx = 0
        res = ''
        for ortho in non_word_ortho:
            list_ortho = list(ortho)
            inp = self.write_to_ex(list_ortho)
            out = self.write_to_ex(['-'] * 100)
            word = self.get_name(list_ortho, self.non_words_ortho)
            name = f"{str(idx)}_{word}_dataSeed-{self.seed}_nonword"
            res += 'name: ' + \
                '{' + name + '}' + \
                '\n' + inp + '\n' + out + ';\n'
            idx += 1
        self.nonword_ex_file_buf = res

        f = open("../data/ex/nonwords" + ".ex", "w")
        f.write(self.nonword_ex_file_buf)

    def write_to_ex(self, row):
        if len(row) == 100:  # target
            data_type = "T: "
        elif len(row) == 18:  # input
            data_type = "I: "
        else:
            print(len(row))
        return data_type + " ".join(map(str, row))

    def get_name(self, inp, words):
        for key in words.keys():
            d = np.sum(abs(np.asarray(words[key]) - np.asarray(inp)))
            if d == 0:
                return key
        raise Exception

    def shuffle(self, arrays):
        prev_len = len(arrays[0])
        p = np.random.permutation(prev_len)
        ret = []
        for array in arrays:
            assert prev_len == len(array)
            prev_len = len(array)
            ret.append(array[p])
        return ret


dg = DataGenerator(seed=0)


# min = -1
# list_dgY = list(dg.y)
# while len(list_dgY) > 0:
#     inp = list_dgY.pop()
#     d = dg.similarity(inp, list_dgY, 'y')
#     if d < min:
#         min = d
# print(min)

# min = -1
# list_dgX = list(dg.X)
# while len(list_dgX) > 0:
#     inp = list_dgX.pop()
#     d = dg.similarity(inp, list_dgX, 'x')
#     if d < min:
#         min = d
# print(min)

min = math.inf
list_dgX = list(dg.words_ortho.values())
while len(list_dgX) > 0:
    inp = list_dgX.pop()
    d = dg.similarity(inp, list_dgX, 'x')
    if d < min:
        min = d
print(min)

# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# word = "computer"
# result_num = 5

# data = []
# target_index = word_to_index[word]
# for i in range(word_embeddings_array.shape[0]):
#     if i != target_index:
#         data.append((index_to_word[i],cosine_similarity([word_embeddings_array[target_index]],[word_embeddings_array[i]])[0][0]))
# data.sort(key=lambda tup: tup[1], reverse=True)
# print (data[:result_num])



import os 
import sys
import logging
import argparse
import numpy as np
from collections import defaultdict
from scipy import linalg, mat, dot, stats
DATA_ROOT = os.path.dirname( os.path.abspath( __file__ ) ) + "/data/"

class Wordsim:
    def __init__(self,lang):
        logging.info("collecting datasets ..")
        self.files = [ file_name.replace(".txt","") for file_name in os.listdir(DATA_ROOT+lang) if ".txt" in file_name ]
        self.dataset=defaultdict(list)
        for file_name in self.files:
            for line in open(DATA_ROOT + lang + "/" + file_name + ".txt"):
                self.dataset[file_name].append([ float(w) if i == 2 else w for i, w in enumerate(line.strip().split())])

    @staticmethod
    def cos(vec1,vec2):
        return vec1.dot(vec2)/(linalg.norm(vec1)*linalg.norm(vec2))

    @staticmethod
    def rho(vec1,vec2):
        return stats.stats.spearmanr(vec1, vec2)[0]

    @staticmethod
    def load_vector(path):
        try:
            logging.info("loading vector ..")
            assert (path[-4:] == ".npz")
            data = np.load(path)
        except ValueError:
            print ("Oops!  No such file.  Try again ..")
        word_embeddings_array = data['word_embeddings_array']
        word_to_index = data['word_to_index'].item()
        index_to_word = data['index_to_word'].item()

        word2vec = {}
        for word in word_to_index: 
            # word = word.decode("utf-8") 
            word2vec[word] = word_embeddings_array[word_to_index[word]]
        logging.info("loaded vector {0} words found ..".format(len(word2vec.keys())))
        return word2vec

    @staticmethod
    def pprint(result):
        from prettytable import PrettyTable
        x = PrettyTable(["Dataset", "Found", "Not Found", "Score (rho)"])
        x.align["Dataset"] = "l"
        for k, v in result.items():
            x.add_row([k,v[0],v[1],v[2]])
        print (x)

    def evaluate(self,word_dict):
        result = {}
        vocab = word_dict.keys()
        # for i in vocab:
        #     print (i)
        for file_name, data in self.dataset.items():
            pred, label, found, notfound = [] ,[], 0, 0
            for datum in data:
                if datum[0] in vocab and datum[1] in vocab:
                    found += 1
                    # print (word_dict[datum[0]],word_dict[datum[1]])
                    pred.append(self.cos(word_dict[datum[0]],word_dict[datum[1]]))
                    label.append(datum[2])
                else:
                    notfound += 1
            result[file_name] = (found, notfound, self.rho(label,pred)*100)
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', '-l', default="en")
    parser.add_argument('--vector', '-v', default="")
    args = parser.parse_args()
    wordsim = Wordsim(args.lang)
    word2vec = wordsim.load_vector(args.vector)
    result = wordsim.evaluate(word2vec)
    wordsim.pprint(result)
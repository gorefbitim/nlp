import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer  # count words
from sklearn.feature_extraction.text import TfidfTransformer # normalize by document frequency
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import metrics

configuration_fn = 'nlp.yml'
results_dir = 'results'

def load_data(fn):
    with open(fn):
        yield fn.readline()

def cleanup(doc):
    digits = [str(d) for d in range(10)]
    pattern_digits = re.compile('|'.join(digits))
    return pattern_digits.sub('', doc.lower())

def train(X_train, y_train, random_seed):
    model = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer())
                   ])
    model.fit(X_train, y_train)

    return model 

def sentences_to_indices(X, word_to_index, max_len):

    m = X.shape[0]  # number of training examples

    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples
        sentence_words = [word.lower().replace('\t', '') for word in X[i].split(' ') if word.replace('\t', '') != '']
        j = 0

        for w in sentence_words[:max_len]:
            X_indices[i, j] = word_to_index.get(w, 0)
            j += 1

    return X_indices

def print_interesting_words(pipeline, count=100):
    classifer  = pipeline[2] 
    features   = classifer.coef_[0]
    vocabulary = pipeline[0].vocabulary_
    threshold  = sorted(list(features))[-count]
    revdict = dict((vocabulary[k],k) for k in iter(vocabulary))
    print([revdict[i] for i,x in enumerate(features) if x>threshold][30:80])

def display_roc_curve(pipeline, X_test, y_test):
    y_score = pipeline.decision_function(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

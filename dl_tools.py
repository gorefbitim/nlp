from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding

def build_model(num_words, embed_dim, input_length, lstm_out):
    model = Sequential()
    model.add(Embedding(num_words, embed_dim, input_length=input_length))
    model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    return model

def build_w2v_model(word_to_vec_map, word_to_index, maxLen):
    model_w2v = Sequential()
    model_w2v.add(pretrained_embedding_layer(word_to_vec_map, word_to_index, maxLen))
    model_w2v.add(LSTM(256, return_sequences=True))
    model_w2v.add(Dropout(0.8))
    model_w2v.add(LSTM(64, return_sequences=False))
    model_w2v.add(Dropout(0.8))
    model_w2v.add(Dense(256, activation=None))
    model_w2v.add(Dropout(0.8))
    model_w2v.add(Dense(60, activation=None))
    model_w2v.add(Dropout(0.8))
    model_w2v.add(Dense(2, activation=None))
    model_w2v.add(Activation('softmax'))
    model_w2v.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], )
    return model_w2v

def tsne_plot_similar_words(title, labels, embedding_clusters, words, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, word, color in zip(labels, embedding_clusters, [words], colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=16)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def pretrained_embedding_layer(word_to_vec_map, word_to_index, maxLen):
    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False, input_shape=(maxLen,))

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def gen(file_name, batchsz = 64):
    csvfile = open(file_name)
    reader = csv.reader(csvfile)
    batchCount = 0
    while True:
        for line in reader:
            inputs = []
            targets = []
            temp_image = cv2.imread(line[1]) # line[1] is path to image
            measurement = line[3] # steering angle
            inputs.append(temp_image)
            targets.append(measurement)
            batchCount += 1
            if batchCount >= batchsz:
                batchCount = 0
                X = np.array(inputs)
                y = np.array(targets)
                yield X, y
        csvfile.seek(0)

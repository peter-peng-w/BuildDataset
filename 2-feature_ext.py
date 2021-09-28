import os
import json
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import _stop_words
punct = string.punctuation


N_FEATURE = 2000
dataset_name = "ratebeer/large_500_pure"


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res


def calTFidf(text):
    # NOTE: STOP Words should be removed
    vectorizer = CountVectorizer(lowercase=True, stop_words='english')
    wordcount = vectorizer.fit_transform(text)
    print("wordcount: ", wordcount.shape)
    tf_idf_transformer = TfidfTransformer()
    tfidf_matrix = tf_idf_transformer.fit_transform(wordcount)
    return vectorizer, tfidf_matrix


if __name__ == "__main__":
    # 0. Read Data
    dir_path = '../Dataset/ratebeer/{}'.format(dataset_name)
    # # Load train dataset
    train_review = []
    cnt = 0
    file_path = os.path.join(dir_path, 'train_review_filtered.json')
    with open(file_path) as f:
        print("Load file: {}".format(file_path))
        for line in f:
            line_data = json.loads(line)
            user_id = line_data['user']
            item_id = line_data['item']
            rating = line_data['rating']
            review = line_data['review']
            train_review.append([item_id, user_id, rating, review])
            cnt += 1
            if cnt % 50000 == 0:
                print('{} lines loaded.'.format(cnt))
    print('Finish loading train dataset, totally {} lines.'.format(len(train_review)))
    # # Load test dataset
    test_review = []
    cnt = 0
    file_path = os.path.join(dir_path, 'test_review_filtered_clean.json')
    with open(file_path) as f:
        print("Load file: {}".format(file_path))
        for line in f:
            line_data = json.loads(line)
            user_id = line_data['user']
            item_id = line_data['item']
            rating = line_data['rating']
            review = line_data['review']
            test_review.append([item_id, user_id, rating, review])
            cnt += 1
            if cnt % 10000 == 0:
                print('{} lines loaded.'.format(cnt))
    print('Finish loading test dataset, totally {} lines.'.format(len(test_review)))
    # # Convert List Data to Pandas Dataframe
    df_train_data = pd.DataFrame(train_review, columns=['item', 'user', 'rating', 'review'])
    df_test_data = pd.DataFrame(test_review, columns=['item', 'user', 'rating', 'review'])
    print("number of user in trainset: {}".format(len(list(df_train_data['user'].unique()))))
    print("number of item in trainset: {}".format(len(list(df_train_data['item'].unique()))))
    # 1. Compute tf-idf value on the whole vocabulary of Train
    # # Get all the reviews on Train-set
    documents = []
    for idx, row in df_train_data.iterrows():
        text = row['review']
        documents.append(text)
    # # Compute tf-idf matrix
    vectorizer, tfidf_matrix = calTFidf(documents)
    print("The number of example is {0}, and the TF-IDF vocabulary size is {1}".format(
        len(documents), len(vectorizer.vocabulary_)))
    # # Compute Mean Tf-idf Score for each word
    word_tfidf = np.array(tfidf_matrix.mean(0))
    # # Sort Wrods According to their mean tf-idf value
    word_order = np.argsort(-word_tfidf[0])
    # 2. Extract the top-N mean tf-idf score words as the feature words
    vocab2id = {}
    id2vocab = {}
    id2word = vectorizer.get_feature_names()
    feature_word_num = N_FEATURE
    feature_word_idx = 0
    cnt = 0
    for idx in word_order:
        # get the word
        word = id2word[idx]
        # check if this word is a number/punctuation
        if word.isdigit() or (word in punct):
            print(word)
        elif word in _stop_words.ENGLISH_STOP_WORDS:
            print('Stop Words: {}'.format(word))
        else:
            # write word into vocab2id mapping
            vocab2id[word] = str(len(vocab2id))
            # write word into id2vocab mapping
            id2vocab[str(cnt)] = word
            cnt += 1
            if cnt == feature_word_num:
                break
    # # Save the feature vocab to json file
    with open('../Dataset/{}/train/feature/feature2id.json'.format(dataset_name), 'w') as f:
        json.dump(vocab2id, f)
    with open('../Dataset/{}/train/feature/id2feature.json'.format(dataset_name), 'w') as f:
        json.dump(id2vocab, f)

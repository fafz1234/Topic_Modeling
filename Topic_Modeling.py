# import dependancies
from __future__ import division
import time, csv, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# import data from file and store on dataframe
main_df = pd.read_csv('/Users/FAVZ/Downloads/Linc_Global_Project/transformed_data.csv', header=None, index_col = False)


# convert data to document matrix
start_time_2 = time.time()
no_features = 1000
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(main_df.iloc[0:, 0])

# split data to train and test set
test_size = 0.0001 # just a small amount of test set to demonstrate model 
train_df, test_df = train_test_split(tf, test_size=test_size, random_state=42)
tf_feature_names = tf_vectorizer.get_feature_names()
time.time() - start_time_2
print 'Time to convert data to document matrix: {} seconds'.format(round(time.time() - start_time_2))

# train model
start_time_3 = time.time()
def get_topics(model, feature_names, no_top_words):
    topics_ = []
    for topic_idx, topic in enumerate(model.components_):
        topics_.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topics_
def get_top_10(topics_,top_10 = 10):
    for i in range(top_10):
        print 'Top 10 most popular topics in the dataset:'
        print 'Topic %d:' % (i+1)
        print topics_[i]
def get_file(file_path, file_name):
    temp_df = pd.read_csv(file_path + '/{}.tsv'.format(file_name), sep='\t',
                quoting=csv.QUOTE_NONE, header=None, index_col = False)
    temp_dialog = pd.Series([temp_df.iloc[0:, temp_df.shape[1] - 1].str.cat(sep = ' ')])
    return temp_dialog

no_topics = 30
# fit LDA model
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online',
                                learning_offset=50.,random_state=0).fit(train_df)
no_top_words = 6
topics_ = get_topics(lda, tf_feature_names, no_top_words)
get_top_10(topics_,10)
time.time() - start_time_3
print 
print 'Time to train model: {} minutes'.format(round((time.time() - start_time_3)/60))

# topic detector (question 2)
"""using the last file the dialoges/4 folder to test the model and generate a set of top 3 relevant topics mentioned 
in the conversation """
new_dialog_path = '/Users/FAVZ/Downloads/Linc_Global_Project/dialogs/4' # path to new conversation file (tsv file)
new_file_name = '269022' # name of the new file

def topic_detector(file_path, file_name):
    transformed_data = get_file(file_path, file_name)
    test_tf = tf_vectorizer.transform(transformed_data)
    predicted = lda.transform(test_tf)
    sort_probabilities = list(*np.argsort(predicted))
    max_probabilities_ind = sort_probabilities[-3:]
    return [str(topics_[i]) for i in max_probabilities_ind]

print
print 'Relevant topics mentioned in conversation'
print topic_detector(new_dialog_path, new_file_name)



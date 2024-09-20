import pandas as pd
import re
from collections import Counter
 


data_file_path = './results/sepsis/2K_pos_samples/'
data_file = 'A41.51_SenTrans_nostopwords_pattern_CountVectorizer_orig.csv'
stopwords_file_path = './expr_datasets/'
stopwords_file = 'icd10_stopwords.csv'

threshold = 0.85
doc_freq = 3
top_keywords = 50

def containsLetterAndNumber(input):
    return input.isalnum() and not input.isalpha() and not input.isdigit()

def remove_stopwords(keyphrases, head_stopwords):
    tail_stopwords = [w for w in head_stopwords if w != '\'s']

    clean_keyphrases = []
    for keyphrase in keyphrases:
        #print('before cleaning: ', keyphrase)
        words = keyphrase[0].split()
        words_to_delete = []
        for word in words:
            if word in head_stopwords or word.isnumeric() or containsLetterAndNumber(word):
                words_to_delete.append(word)
            else:
                break
        for word in reversed(words):
            if word in tail_stopwords or word.isnumeric() or containsLetterAndNumber(word):
                words_to_delete.append(word)
            else:
                break

        for word in words_to_delete:
            if len(words) != 0:
                words.remove(word)
        
        if len(words) != 0:
            words = ' '.join(words)
            keyphrase = (words, keyphrase[1])
            #print('after cleaning: ', keyphrase)
            clean_keyphrases.append(keyphrase)

    return clean_keyphrases

df = pd.read_csv(data_file_path + data_file)
keywords = df.loc[0]['similar_keywords']
similar_keywords = list(eval(keywords))

stopwords_df = pd.read_csv(stopwords_file_path + stopwords_file)
print('stopwords_df shape is: ', stopwords_df.shape)
stopwords_list = stopwords_df['words'].tolist()


#remove the stopwords in the beginning and at the end of a keyphrase
similar_keywords = remove_stopwords(similar_keywords, stopwords_list)


#find the unique keywords
unique_keywords = [t for t in (set(i[0] for i in similar_keywords))]
#print('unique_keywords are:', unique_keywords)

print('Find the max score for each key phrase...')
keyword_val_dict = {}
for unique_keyword in unique_keywords:
    scores = []
    for keyword in similar_keywords:
        if keyword[0] == unique_keyword:
            scores.append(keyword[1])

    keyword_val_dict[unique_keyword] = max(scores)

print('Assign the max score to each key phrase...')
new_similar_keywords = []
for keyword in similar_keywords:
    new_similar_keywords.append((keyword[0], keyword_val_dict[keyword[0]]))
     
#only keep the keywords which have similarity values greater than the threshold
new_similar_keywords = [element for element in new_similar_keywords if element[1] >= threshold]

doc_freqs = Counter(new_similar_keywords)
similar_keywords = []
for new_similar_keyword in new_similar_keywords:
    similar_keywords.append((new_similar_keyword[0], new_similar_keyword[1], doc_freqs[new_similar_keyword]))

#remove the duplicate phrases
similar_keywords = [t for t in (set(i for i in similar_keywords))]

#sort the keywords from high to low
similar_keywords.sort(key=lambda tup:(-tup[1], -tup[2], tup[0]))

#keep the key phrases with certain document frequency
similar_keywords = [element for element in similar_keywords if element[2] >= doc_freq]

#only keep the top keywords
#similar_keywords = similar_keywords[:top_keywords]
print(similar_keywords)


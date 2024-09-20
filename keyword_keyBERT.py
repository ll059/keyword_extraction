import pandas as pd
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer
#from transformers.pipelines import pipeline

file_path = './datasets/sepsis/'
clients = ['advctz']

sentence_model = SentenceTransformer("all-mpnet-base-v2")
kw_model = KeyBERT(model=sentence_model)

def extract_terms(document, n_gram_range = (3,3), 
                  top_N=40, model=kw_model, diversity_threshold = 0.7):

    keywords = model.extract_keywords(docs=document, 
                                    stop_words='english',
                                    #keyphrase_ngram_range=n_gram_range,
                                    #use_mmr=True, 
                                    #diversity = diversity_threshold,
                                    vectorizer=KeyphraseCountVectorizer(),
                                    top_n=top_N)
    
    #return sorted(keywords, key=lambda tup:(-tup[1], tup[0]))
    return keywords

key_df = pd.DataFrame(columns=['visitId', 'keywords'])
for client in clients:
    df = pd.read_csv(file_path + '{0}_dementia_adult_notes.csv'.format(client)) 
    print('df shape is: ', df.shape)

    count = 0
    #for notes in df["documents"].tolist():
    for index in df.index:
        notes = df.loc[index]['documents']
        p_id = df.loc[index]['visitId']
        print("Patient ID:{}\n--> Patient ID counts: {}\n--> length: {}".format(p_id, count, len(notes.split())))
        keywords = extract_terms(notes)
        #print(keywords)
       
        key_df.loc[len(key_df.index)] = [p_id, keywords] 
        count +=1
        print("-"*30)

out_path = './results/'
key_df.to_csv(out_path + 'extraced_keywords_sepsis.csv', index=False)
 
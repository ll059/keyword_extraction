import pandas as pd
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline
from itertools import islice

file_path = '/home/james/Projects/keyword_extraction/datasets/sepsis/'
pos_sepsis_file = 'pos_2000_sepsi_notes.csv'
icd10_file = 'temp_icd-10_code_descript.csv'

threshold = 0.5
top_keywords = 50

#Method 1: sentence transformers: all-mpnet-base-v2
#sentence_model = SentenceTransformer("all-mpnet-base-v2")
#kw_model = KeyBERT(model=sentence_model)
#llm_type = 'sentence_transformer' 

#Method 2: Huggingface model:gatortron_base with 345 million parameters
hf_model = pipeline("feature-extraction", model="hf_models/gatortron_base", device=0)
#hf_model = pipeline("feature-extraction", model='hf_models/mimiciii_bert_10e_128b', device=0)
kw_model = KeyBERT(model=hf_model)
llm_type = 'huggingface_transformer'

def extract_terms(document, pid, icd10_desc, llm_type, n_gram_range = (3,3), 
                  top_N=10, model=kw_model, diversity_threshold = 0.7):

    keywords = model.extract_keywords(docs=document,
                                      p_id=pid,
                                      icd10_descript=icd10_desc,
                                      model_type=llm_type,
                                      #stop_words='english',
                                      #keyphrase_ngram_range=n_gram_range,
                                      #use_mmr=True, 
                                      #diversity = diversity_threshold,
                                      vectorizer=KeyphraseCountVectorizer(),
                                      top_n=top_N)
    
    return keywords

note_df = pd.read_csv(file_path + pos_sepsis_file) 
icd10_df = pd.read_csv(file_path + icd10_file)

for idx in icd10_df.index:
    icd10_code = icd10_df.loc[idx]['code']
    icd10_description = icd10_df.loc[idx]['description']
    print('Processing {}: {}'.format(icd10_code, icd10_description))
    
    key_df = pd.DataFrame(columns=['icd10_code', 'descriptions', 'similar_keywords'])
    similar_keywords = []
     
    notes = note_df['documents'].tolist()
    p_id = note_df['visitId'].tolist()
    sublist_len = 50
    length_to_split = [sublist_len for i in range(int(len(p_id)/sublist_len))]
    if len(p_id)%sublist_len != 0:
        length_to_split.append(len(p_id)%sublist_len)
    
    notes = iter(notes)
    notes = [list(islice(notes, elem)) for elem in length_to_split]  
    p_id = iter(p_id)
    p_id = [list(islice(p_id, elem)) for elem in length_to_split]
    
    
    for index, _ in enumerate(p_id):
        print('Batch index is: ', index)
        keywords = extract_terms(notes[index], p_id[index], icd10_description, llm_type)
        for item in keywords:
            similar_keywords.append(item)
    
    
    similar_keywords = [element for sublist in similar_keywords for element in sublist]
    similar_keywords = [element for element in similar_keywords if element[1] >= threshold]
    similar_keywords = [t for t in (set(tuple(i) for i in similar_keywords))]
    similar_keywords.sort(key=lambda tup:(-tup[1], tup[0]))
    similar_keywords = similar_keywords[:top_keywords]
    print('similar_keywords are: ', similar_keywords)
    key_df.loc[len(key_df.index)] = [icd10_code, icd10_description, similar_keywords] 
    out_path = './results/gatortron-base_results/'
    key_df.to_csv(out_path + '{}_keywords.csv'.format(icd10_code), index=False)
    
 
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from typing import List, Union, Tuple

from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer 
 
from keybert.backend._utils import select_backend


class MMR:
    """
    Calculate the Maximal Marginal Relevance
    """

    def __init__(self, model="all-MiniLM-L6-v2"):
        """KeyBERT initialization

        Arguments:
            model: Use a custom embedding model.
                   The following backends are currently supported:
                      * SentenceTransformers
                      * 🤗 Transformers
                      * Flair
                      * Spacy
                      * Gensim
                      * USE (TF-Hub)
                    You can also pass in a string that points to one of the following
                    sentence-transformers models:
                      * https://www.sbert.net/docs/pretrained_models.html
        """
        self.model = select_backend(model)

    def calc_mmr(
        self,
        words: Union[str, List[str]],
        icd10_descript: str = '',
        model_type: str = 'sentence_transformer',
        diversity: float = 0.5,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract keywords and/or keyphrases

        To get the biggest speed-up, make sure to pass multiple documents
        at once instead of iterating over a single document.

        Arguments:
            words: The document(s) for which to extract keywords/keyphrases
            top_n: Return the top n keywords/keyphrases
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    NOTE: This is not used if you passed a `vectorizer`.
            use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases.
            diversity: The diversity of the results between 0 and 1 if `use_mmr`
                       is set to True.

        Returns:
            keywords: The top n keywords for a document with their respective distances
                      to the input document.

        Usage:

        To extract keywords from a single document:

        ```python
        from keybert import MMR

        kw_model = MMR()
        keywords = kw_model.calc_mmr()
        ```
        """
        
        # Check for a single, empty document
        if isinstance(words, str):
            if words:
                words = [words]
            else:
                return []
 
        if model_type == 'huggingface_transformer':
            words = np.append(words, icd10_descript)

        # Extract embeddings
        print('Computing embeddings of key phrases and ICD10 definition!!!')
        word_embeddings = self.model.embed(words)
        if model_type == 'sentence_transformer':
            icd10_embeddings = self.model.embed(icd10_descript)
        elif model_type == 'huggingface_transformer':
            icd10_embeddings = word_embeddings[-1]
        icd10_embeddings = icd10_embeddings.reshape(1, -1)

        keywords = []
        try:
            print('Computing MMR...') 
            
            # Extract similarity within words, and between words and the icd10 definition
            word_icd10_similarity = cosine_similarity(word_embeddings, icd10_embeddings)
            word_similarity = cosine_similarity(word_embeddings)

            # Initialize candidates and already choose best keyword/keyphras
            keywords_idx = [np.argmax(word_icd10_similarity)]
            candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

            for _ in range(len(words) - 1):
                # Extract similarities within candidates and
                # between candidates and selected keywords/phrases
                candidate_similarities = word_icd10_similarity[candidates_idx, :]
                target_similarities = np.max(
                        word_similarity[candidates_idx][:, keywords_idx], axis=1)

                # Calculate MMR
                mmr = (
                       1 - diversity
                       ) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
                mmr_idx = candidates_idx[np.argmax(mmr)]

                # Update keywords & candidates
                keywords.append((words[mmr_idx], max(mmr)))
                candidates_idx.remove(mmr_idx)
    
        # Capturing empty keywords
        except ValueError:
            keywords.append([])

        return keywords

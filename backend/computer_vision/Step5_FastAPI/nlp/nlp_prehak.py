
import pandas as pd
import numpy as np
import re
import string
import multiprocessing as mp
import spacy

# Load the English model

from functools import reduce
from operator import add
from sklearn.feature_extraction.text import TfidfVectorizer

from itertools import repeat




# Set All Recommendation Model Parameters
N_topics = 50             # Number of Topics to Extract from corpora
N_top_docs = 200          # Number of top documents within each topic to extract keywords
N_top_words = 25          # Number of keywords to extract from each topic
N_docs_categorized = 2000 # Number of top documents within each topic to tag
N_neighbor_window = 4     # Length of word-radius that defines the neighborhood for
                          # each word in the TextRank adjacency table

# Query Similarity Weights
w_title = 0.2
w_text = 0.3
w_categories = 0.5
w_array = np.array([w_title, w_text, w_categories])

# Query Similarity Weights
w_title = .2
w_text = .3
w_categories = .5



# Recipe Stopwords: for any high volume food recipe terminology that doesn't contribute
# to the searchability of a recipe. This list must be manually created.
recipe_stopwords = ['cup','cups','ingredient','ingredients','teaspoon','teaspoons','tablespoon',
                   'tablespoons','C','F']

# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


recipes=None
text_tfidf=None
title_tfidf=None
tags_tfidf=None
all_text=None
root_text_data=None
vectorizer=None

nlp=None

def init_nlp():
    global  recipes, text_tfidf, title_tfidf, tags_tfidf,all_text, root_text_data
    global vectorizer
    global nlp
    recipes = pd.read_pickle('./nlp/preprocessed_recipes.pkl')


    ingredients = []
    for ing_list in recipes['ingredients']:
        clean_ings = [ing.replace('ADVERTISEMENT', '').strip() for ing in ing_list]
        if '' in clean_ings:
            clean_ings.remove('')
        ingredients.append(clean_ings)
    recipes['ingredients'] = ingredients

    recipes['ingredient_text'] = ['; '.join(ingredients) for ingredients in recipes['ingredients']]
    recipes['ingredient_text'].head()

    recipes['ingredient_count'] = [len(ingredients) for ingredients in recipes['ingredients']]

    all_text = recipes['title'] + ' ' + recipes['ingredient_text'] + ' ' + recipes['instructions']


    cleaned_text = clean_text(all_text)

    # Testing Strategies and Code
    nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"]) # TODO Added on 3103-0457 , disable=["parser", "ner"]
    ' '.join([token.lemma_ for token in nlp(cleaned_text[2]) if not token.is_stop])


    root_text_data = cleaned_text


    recipes = pd.read_csv('./nlp/tagged_recipes_df.csv')


    recipes['tag_list'] = recipes['tag_list'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

    # Concatenate lists of tags into a string for each document
    recipes['tags'] = [' '.join(tags) for tags in recipes['tag_list']]


    # Creating TF-IDF Matrices and recalling text dependencies

    tokenized_text = pd.read_csv('./nlp/tokenized_text.csv')

    # Extracting the tokenized text from the DataFrame
    tokenized_text = tokenized_text['0']
    # new
    # Create the TfidfVectorizer
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))
    # Fit and transform your data
    text_tfidf = vectorizer.fit_transform(tokenized_text)
    # Get feature names
    tfidf_words = vectorizer.get_feature_names_out()
    print(text_tfidf.shape)
    print(len(tfidf_words))




    # Fit and transform your data
    text_tfidf = vectorizer.fit_transform(tokenized_text)
    title_tfidf = vectorizer.transform(recipes['title'])
    # text_tfidf    <== Variable with recipe ingredients and instructions
    tags_tfidf = vectorizer.transform(recipes['tags'])
    # recipes   <== DataFrame; For indexing and printing recipes

    return recipes, text_tfidf, title_tfidf, tags_tfidf,all_text

def clean_text(documents):
    cleaned_text = []
    for doc in documents:
        doc = doc.translate(str.maketrans('', '', string.punctuation)) # Remove Punctuation
        doc = re.sub(r'\d+', '', doc) # Remove Digits
        doc = doc.replace('\n',' ') # Remove New Lines
        doc = doc.strip() # Remove Leading White Space
        doc = re.sub(' +', ' ', doc) # Remove multiple white spaces
        cleaned_text.append(doc)
    return cleaned_text

def text_tokenizer_mp(doc):
    tok_doc = ' '.join([token.lemma_ for token in nlp(doc) if not token.is_stop])
    return tok_doc

def topic_docs_4kwsummary(topic_document_scores, root_text_data):
    '''Gathers and formats the top recipes in each topic'''
    text_index = pd.Series(topic_document_scores).sort_values(ascending = False)[:N_top_docs].index
    text_4kwsummary = pd.Series(root_text_data)[text_index]
    return text_4kwsummary

def generate_filter_kws(text_list):
    '''Filters out specific parts of speech and stop words from the list of potential keywords'''
    parsed_texts = nlp(' '.join(text_list))
    kw_filts = set([str(word) for word in parsed_texts
                if (word.pos_== ('NOUN' or 'ADJ' or 'VERB'))
                and word.lemma_ not in recipe_stopwords])
    return list(kw_filts), parsed_texts

def generate_adjacency(kw_filts, parsed_texts):
    '''Tabulates counts of neighbors in the neighborhood window for each unique word'''
    adjacency = pd.DataFrame(columns=kw_filts, index=kw_filts, data = 0)
    for i, word in enumerate(parsed_texts):
        if any ([str(word) == item for item in kw_filts]):
            end = min(len(parsed_texts), i+N_neighbor_window+1) # Neighborhood Window Utilized Here
            nextwords = parsed_texts[i+1:end]
            inset = [str(x) in kw_filts for x in nextwords]
            neighbors = [str(nextwords[i]) for i in range(len(nextwords)) if inset[i]]
            if neighbors:
                adjacency.loc[str(word), neighbors] += 1
    return adjacency


def generate_kw_index(topic_document_scores):
    kw_index = pd.Series(topic_document_scores).sort_values(ascending = False)[:N_docs_categorized].index
    return kw_index

def qweight_array(query_length, qw_array = [1]):
    '''Returns descending weights for ranked query ingredients'''
    if query_length > 1:
        to_split = qw_array.pop()
        split = to_split/2
        qw_array.extend([split, split])
        return qweight_array(query_length - 1, qw_array)
    else:
        return np.array(qw_array)

def ranked_query(query):
    '''Called if query ingredients are ranked in order of importance.
    Weights and adds each ranked query ingredient vector.'''
    query = [[q] for q in query]      # place words in seperate documents
    q_vecs = [vectorizer.transform(q) for q in query]
    qw_array = qweight_array(len(query),[1])
    q_weighted_vecs = q_vecs * qw_array
    q_final_vector = reduce(np.add,q_weighted_vecs)
    return q_final_vector

def overall_scores(query_vector):
    '''Calculates Query Similarity Scores against recipe title, instructions, and keywords.
    Then returns weighted averages of similarities for each recipe.'''
    final_scores = title_tfidf*query_vector.T*w_title
    final_scores += text_tfidf*query_vector.T*w_text
    final_scores += tags_tfidf*query_vector.T*w_categories
    return final_scores


class Recipe:
    def __init__(self, title, ingredients, instructions, rank):
        self.title = title
        self.ingredients = ingredients
        self.instructions = instructions
        self.rank = rank

    def __str__(self):
        # This method returns a string representation of the Recipe object
        return (f"Recipe Rank: {self.rank}\n"
                f"Title: {self.title}\n"
                f"Ingredients:\n{self.ingredients}\n"
                f"Instructions:\n{self.instructions}\n")

    def __repr__(self):
        return f"Recipe Rank: {self.rank}\nTitle: {self.title}\nIngredients: {self.ingredients}\nInstructions: {self.instructions}\n"


def print_recipes(index, query, recipe_range):
    '''Prints recipes according to query similarity ranks and saves them as Recipe objects.'''

    recipe_objects = {}
    print('Search Query: {}\n'.format(query))

    for i, idx in enumerate(index, start=recipe_range[0]):
        title = recipes.loc[idx, 'title']
        ingredients = recipes.loc[idx, 'ingredient_text']
        instructions = recipes.loc[idx, 'instructions']
        rank = i + 1

        # Create a Recipe object and store it in the dictionary with rank as the key
        recipe_obj = Recipe(title, ingredients, instructions, rank)
        recipe_objects[rank] = recipe_obj

        # This will print the recipe using the __repr__ method of the Recipe class
        print(recipe_obj)

    return recipe_objects



def Search_Recipes(query, query_ranked=False, recipe_range=(0,3)):
    '''Master Recipe Search Function'''
    init_nlp()
    if query_ranked == True:
        q_vector = ranked_query(query)
    else:
        q_vector = vectorizer.transform([' '.join(query)])
    recipe_scores = overall_scores(q_vector)
    sorted_index = pd.Series(recipe_scores.toarray().T[0]).sort_values(ascending = False)[recipe_range[0]:recipe_range[1]].index


    return print_recipes(sorted_index, query, recipe_range)

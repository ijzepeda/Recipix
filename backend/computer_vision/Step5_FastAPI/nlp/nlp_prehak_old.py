
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

# AUthors: Mahakdeep KAur, Preet and Ivan Zepeda


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




class NLPModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLPModel, cls).__new__(cls)
            cls._initialize()
        return cls._instance

    @classmethod
    def _initialize(cls):
        # Load pickles and CSVs
        cls.recipes = pd.read_pickle('./nlp/preprocessed_recipes.pkl')

        # Clean ingredients
        ingredients = []
        for ing_list in cls.recipes['ingredients']:
            clean_ings = [ing.replace('ADVERTISEMENT', '').strip() for ing in ing_list]
            if '' in clean_ings:
                clean_ings.remove('')
            ingredients.append(clean_ings)
        cls.recipes['ingredients'] = ingredients

        cls.recipes['ingredient_text'] = ['; '.join(ingredients) for ingredients in cls.recipes['ingredients']]
        cls.recipes['ingredient_count'] = [len(ingredients) for ingredients in cls.recipes['ingredients']]

        cls.all_text = cls.recipes['title'] + ' ' + cls.recipes['ingredient_text'] + ' ' + cls.recipes['instructions']

        # Initialize NLP model
        cls.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])

        cls.recipes_csv = pd.read_csv('./nlp/tagged_recipes_df.csv')
        cls.recipes_csv['tag_list'] = cls.recipes_csv['tag_list'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        cls.recipes_csv['tags'] = [' '.join(tags) for tags in cls.recipes_csv['tag_list']]

        # Creating TF-IDF Matrices
        tokenized_text = pd.read_csv('./nlp/tokenized_text.csv')['0']
        cls.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))
        cls.text_tfidf = cls.vectorizer.fit_transform(tokenized_text)
        cls.title_tfidf = cls.vectorizer.transform(cls.recipes_csv['title'])
        cls.tags_tfidf = cls.vectorizer.transform(cls.recipes_csv['tags'])

def init_nlp():
    nlp_model = NLPModel()  # This will initialize the model and data if not already done
    return nlp_model #este no estaba
    # return nlp_model.recipes, nlp_model.text_tfidf, nlp_model.title_tfidf, nlp_model.tags_tfidf, nlp_model.all_text

# def init_nlp():
#     global  recipes, text_tfidf, title_tfidf, tags_tfidf,all_text, root_text_data
#     global vectorizer
#     global nlp
#     recipes = pd.read_pickle('./nlp/preprocessed_recipes.pkl')
#
#
#     ingredients = []
#     for ing_list in recipes['ingredients']:
#         clean_ings = [ing.replace('ADVERTISEMENT', '').strip() for ing in ing_list]
#         if '' in clean_ings:
#             clean_ings.remove('')
#         ingredients.append(clean_ings)
#     recipes['ingredients'] = ingredients
#
#     recipes['ingredient_text'] = ['; '.join(ingredients) for ingredients in recipes['ingredients']]
#     recipes['ingredient_text'].head()
#
#     recipes['ingredient_count'] = [len(ingredients) for ingredients in recipes['ingredients']]
#
#     all_text = recipes['title'] + ' ' + recipes['ingredient_text'] + ' ' + recipes['instructions']
#
#
#     cleaned_text = clean_text(all_text)
#
#     # Testing Strategies and Code
#     nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"]) # TODO Added on 3103-0457 , disable=["parser", "ner"]
#     ' '.join([token.lemma_ for token in nlp(cleaned_text[2]) if not token.is_stop])
#
#
#     root_text_data = cleaned_text
#
#
#     recipes = pd.read_csv('./nlp/tagged_recipes_df.csv')
#
#
#     recipes['tag_list'] = recipes['tag_list'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
#
#     # Concatenate lists of tags into a string for each document
#     recipes['tags'] = [' '.join(tags) for tags in recipes['tag_list']]
#
#
#     # Creating TF-IDF Matrices and recalling text dependencies
#
#     tokenized_text = pd.read_csv('./nlp/tokenized_text.csv')
#
#     # Extracting the tokenized text from the DataFrame
#     tokenized_text = tokenized_text['0']
#     # new
#     # Create the TfidfVectorizer
#     vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))
#     # Fit and transform your data
#     text_tfidf = vectorizer.fit_transform(tokenized_text)
#     # Get feature names
#     tfidf_words = vectorizer.get_feature_names_out()
#     print(text_tfidf.shape)
#     print(len(tfidf_words))
#
#
#
#
#     # Fit and transform your data
#     text_tfidf = vectorizer.fit_transform(tokenized_text)
#     title_tfidf = vectorizer.transform(recipes['title'])
#     # text_tfidf    <== Variable with recipe ingredients and instructions
#     tags_tfidf = vectorizer.transform(recipes['tags'])
#     # recipes   <== DataFrame; For indexing and printing recipes
#
#     return recipes, text_tfidf, title_tfidf, tags_tfidf,all_text

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




def sort_ingredients(ingredients):
    # Normalize all ingredient names to lowercase
    ingredients = [ingredient.lower() for ingredient in ingredients]

    # Define the priority for each category
    priority = {
        'proteins': 1,
        'special': 2,
        'dairy': 2,
        'grains and staples': 3,
        'vegetables': 4,
        'fruits': 5,
        'spices, herbs, and others': 6,
    }

    # Map each ingredient to its category, also in lowercase
    category_map = {
        # Proteins
        'chicken': 'proteins',  'ground meat': 'proteins',
        'meat': 'proteins',  'salmon': 'proteins', 'beef meat': 'proteins', 'fish': 'proteins',
        # Special
        'eggs': 'special', 'paneer': 'special', 'ham': 'special', 'bacon': 'special',  # protein
        'kimchi': 'special',  # vegetables

        # Dairy
        'butter': 'dairy', 'cheese': 'dairy', 'milk': 'dairy', 'yogurt': 'dairy',

        # Grains and Staples
        'beans': 'grains and staples', 'black beans': 'grains and staples', 'bread': 'grains and staples',
        'chickpeas': 'grains and staples', 'corn': 'grains and staples', 'lentils': 'grains and staples',
        'pasta': 'grains and staples', 'rice': 'grains and staples', 'soy beans': 'grains and staples',
        'spaghetti': 'grains and staples',

        # Vegetables
        'artichoke': 'vegetables', 'avocado': 'vegetables', 'beetroot': 'vegetables', 'bitter gourd': 'vegetables',
        'broccoli': 'vegetables', 'cabbage': 'vegetables', 'cauliflower': 'vegetables', 'cucumber': 'vegetables',
        'eggplant': 'vegetables', 'garlic': 'vegetables', 'gourd': 'vegetables', 'mushroom': 'vegetables',
        'onion': 'vegetables', 'radish': 'vegetables', 'sweet potato': 'vegetables', 'tomato': 'vegetables',
        'turnip': 'vegetables', 'asparagus': 'vegetables', 'bell pepper': 'vegetables', 'cambray': 'vegetables',
        'carrots': 'vegetables', 'celery': 'vegetables', 'chayote': 'vegetables', 'green beans': 'vegetables',
        'jalapeno': 'vegetables', 'lettuce': 'vegetables', 'okra': 'vegetables',
        'potatoes': 'vegetables', 'spinach': 'vegetables', 'zucchini': 'vegetables',

        # Fruits
        'banana': 'fruits', 'orange': 'fruits', 'papaya': 'fruits', 'apple': 'fruits', 'blueberries': 'fruits',
        'cantaloupe': 'fruits', 'cherries': 'fruits', 'grapefruit': 'fruits', 'grapes': 'fruits', 'guava': 'fruits',
        'kiwi': 'fruits', 'lemon': 'fruits', 'lime': 'fruits', 'mango': 'fruits', 'peach': 'fruits', 'pear': 'fruits',
        'pineapple': 'fruits', 'plums': 'fruits', 'pomegranate': 'fruits', 'raspberries': 'fruits',
        'strawberries': 'fruits', 'tangerine': 'fruits', 'watermelon': 'fruits',

        # Spices, Herbs, and Others
        'cinnamon': 'spices, herbs, and others', 'ginger': 'spices, herbs, and others',
        'chilli pepper': 'spices, herbs, and others', 'cilantro': 'spices, herbs, and others',
        'paprika': 'spices, herbs, and others', 'mac&cheese': 'spices, herbs, and others',
        'juice bottle': 'spices, herbs, and others', 'yam': 'spices, herbs, and others',
        # Assuming 'train' is not an ingredient, it will not be included
        'peas': 'spices, herbs, and others',  # peas can be considered a vegetable too, depends on context
    }

    # Sort the ingredients based on the category priority, handling unknown ingredients
    sorted_ingredients = sorted(
        ingredients,
        key=lambda x: priority.get(category_map.get(x, 'spices, herbs, and others'), 6)
    )

    return sorted_ingredients



# nlp_model=None#este no estaba
def Search_Recipes(query, query_ranked=False, recipe_range=(0,3)):
    '''Master Recipe Search Function'''
    query=sort_ingredients(query)
    # global nlp_model
    # nlp_model = (
    init_nlp()
    if query_ranked == True:
        q_vector = ranked_query(query)
    else:
        q_vector = vectorizer.transform([' '.join(query)])
    recipe_scores = overall_scores(q_vector)
    sorted_index = pd.Series(recipe_scores.toarray().T[0]).sort_values(ascending = False)[recipe_range[0]:recipe_range[1]].index


    return print_recipes(sorted_index, query, recipe_range)

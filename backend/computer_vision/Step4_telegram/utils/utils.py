def sort_ingredients(ingredients):
    """Area of improvement: beign able to change priority based on taste and cravings"""

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
        'chicken': 'proteins', 'ground meat': 'proteins',
        'meat': 'proteins', 'salmon': 'proteins', 'beef meat': 'proteins', 'fish': 'proteins',
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






# Made with <3
# by Ivan Zepeda
# github@ijzepeda-LC
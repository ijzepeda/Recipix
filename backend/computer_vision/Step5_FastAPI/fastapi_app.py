from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
import os
from cv_ingredients import get_ingredients, get_model_config
import time

# from nlp.nlp_prehak import Search_Recipes
from nlp.nlp_prehak import Search_Recipes
from utils.image_dalle import get_dalle_image
import uuid
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# WHISPER
WHISPER_MODEL = "small"
language = "english"
user_data = "static/user_data"  # server folder
file_location = None

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


# import whisper
# whisper_model=None

def init():
    # global whisper_model

    if not os.path.exists(os.path.join('./', (user_data))):
        print(f"Making a folder for {user_data}")
        os.makedirs(os.path.join('./', (user_data)))


# Function to generate a short unique ID
def generate_unique_id():
    return uuid.uuid4().hex[:8]  # Generate an 8-character unique ID


@app.get("/", response_class=HTMLResponse)
async def root():
    with open('templates/index.html', 'r') as html_file:
        return HTMLResponse(content=html_file.read(), status_code=200)


def search_recipes(ingredients):
    print("Searching Recipes")
    rec = Search_Recipes(ingredients, query_ranked=True, recipe_range=(0, 3))  # TODO: TESTING uncomment please

    if rec:
        first_recipe_rank = sorted(rec.keys())[0]
        first_recipe = rec[first_recipe_rank]

        message_text = {
            "title": first_recipe.title,
            "ingredients": first_recipe.ingredients,
            "instructions": first_recipe.instructions,
        }

        image_url = None
        other_recipes_list = [{"rank": rank, "title": rec[rank].title, "ingredients": rec[rank].ingredients,
                               "instructions": rec[rank].instructions} for rank in sorted(rec.keys())[1:]]

        return {
            "top_recipe": message_text,
            "title": first_recipe.title,
            "ingredients": first_recipe.ingredients,
            "instructions": first_recipe.instructions,
            "image_url": image_url,
            "other_recipes": other_recipes_list
        }
    else:
        print("No Recipes found")
        return {"message": "message: No recipes found.",
                "top_recipe": "No recipes",
                "title": "No recipes found",
                "ingredients": "Not enought ingredients",
                "instructions": "No instructions",
                "image_url": "static/imgs/recipix_logo.png",
                "other_recipes": [{"title": "Get more ingredients at Costco or FreshCo"}, ]
                }


@app.post("/upload_photo/")
async def upload_photo(photo: UploadFile = File(...)):
    global file_location
    start_time = time.time()
    # Extract the file extension from the uploaded file
    extension = os.path.splitext(photo.filename)[1]
    # Create a new filename using a portion of the original name and a unique ID
    new_name = f"{photo.filename.split('.')[0][:10]}_{generate_unique_id()}{extension}"

    # Save the uploaded file under the new name
    file_location = f"static/user_data/{new_name}"
    print("Uploaded photo is in ", file_location)

    with open(file_location, "wb+") as file_object:
        file_object.write(await photo.read())

    ingredients = get_ingredients(file_location, detailed=False, debug=False, _CONFIDENCE_THRESH=0.75)
    print("Time to process image to ingredients: ", time.time() - start_time)

    return process_recipe(ingredients)


def process_recipe(ingredients):
    """ The return JSON format is:
    {
        "ingredients": [],
        "JSON": {
            "top_recipe": {
                "title": " ",
                "ingredients": "",
                "instructions": ""
            },
            "other_recipes": [
                {
                    "rank": 2,
                    "title": "",
                    "ingredients": "",
                    "instructions": ""
                },
                {
                    "rank": 3,
                    "title": "",
                    "ingredients": "",
                    "instructions": ""
                }
            ]
        }
    }
    """

    start_time = time.time()
    recipes = search_recipes(ingredients)
    print("Time to process recipes with ingredients: ", time.time() - start_time)

    # DAlle image
    start_time = time.time()
    image_url = None
    try:
        image_url = get_dalle_image(recipes['title'], ingredients)
        recipes["image_url"] = image_url
    except Exception as e:
        print("Creating image Failed")
        print("Error generating Image with Dall-e", e)
        print("This was the prompt", recipes['title'], recipes['ingredients'])
    print("Time to process DAlle image with recipe: ", time.time() - start_time)

    eljason = {"ingredients": ingredients, "JSON": recipes}

    return eljason


@app.post("/upload_text/")
async def upload_text(text: str = Form(...)):
    # Here, you would process and possibly store the received text.
    ingredients = text.split(',')
    return process_recipe(ingredients)


@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    # Here, you would save the audio file and possibly process it.
    return {"filename": file.filename}

# Made with <3
# by Ivan Zepeda
# github@ijzepeda-LC

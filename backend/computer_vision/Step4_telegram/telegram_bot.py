#!/usr/bin/env python

import logging
import random
import json

from telegram import __version__ as TG_VER, InlineKeyboardMarkup, InlineKeyboardButton
import whisper
import openai  # New version of whisper available on openai
# from openai import whisper
import os
import toml
from telegram import ReplyKeyboardRemove, Update

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import MessageHandler, filters, ConversationHandler

from utils.utils import sort_ingredients
from cv_ingredients import get_ingredients, get_model_config
import time

from nlp.nlp_prehak import Search_Recipes, Search_Recipes_filtering
from utils.image_dalle import get_dalle_image

from utils.to_pdf import save_pdf_style2

# ------- VARIABLES, PATHS & CONSTANTS ------

TOKEN = toml.load('./secrets.toml')['TELEGRAM_API_KEY']

verbose = True
generate_photo = False

# Define the different states a chat can be in
VOICE, AUDIO, TRANSLATE, PHOTO, ALLERGIES = range(5)
BOTNAME = "Recipix"
WHISPER_MODEL = "small"
language = "english"
user_data = "user_data"  # server folder

USER_PREFS_JSON_PATH = "./user_data/user_prefs.json"

# TELEGRAM config vers

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 5):
    raise RuntimeError(
        f"This bot is not compatible with your current PTB version {TG_VER}. Please use a different device Sorry!"

    )

# TELEGRAM INITIALIZATION
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

if not os.path.exists(os.path.join('./', (user_data))):
    print(f"Making a folder for {user_data}")
    os.makedirs(os.path.join('./', (user_data)))

try:
    model = whisper.load_model(WHISPER_MODEL, device='cuda')  # or gpu
    print("GPU Found,")
except:
    print("No GPU found, using CPU")
    model = whisper.load_model(WHISPER_MODEL, device='cpu')


def get_transcript(audio_file):
    out = model.transcribe(audio_file)
    return out['text']


def load_user_data(username):
    # Create a singleton? is this per user? for projects sake, one file consider all
    # inside user_data/users_prefs.json with { user:username {}}
    print("Loading user data")
    with open('./user_data/user_prefs.json') as f:
        data = json.load(f)
        return data.get(username, None)


# ----------------------------- Async Functions ---------------------------------------------

def split_message(text, max_length=4000):
    """Splits a text message into parts where each part has a maximum length | Original 4096 characters."""
    parts = []
    while len(text) > 0:
        # Find the maximum message size that's less than the max_length
        # or the remainder of the message if it's shorter than max_length
        if len(text) > max_length:
            # Try to split at newline to avoid breaking a line
            split_at = text.rfind('\n', 0, max_length)
            if split_at == -1:  # No newline found; split at max_length
                split_at = max_length
            part = text[:split_at].strip()
            text = text[split_at:].strip()
        else:
            part = text
            text = ""
        parts.append(part)
    return parts


async def send_split_message(update, message_text):
    """Splits a long message into parts and sends each part separately."""
    message_parts = split_message(message_text)
    for part in message_parts:
        await update.message.reply_text(part)


async def ensure_user_folder_exists(user):
    folder_path = os.path.join('./user_data/', str(user.first_name))
    if not os.path.exists(folder_path):
        print(f"Making a folder for {str(user.first_name)}")
        os.makedirs(folder_path)


async def process_recipes_and_respond(ingredients, update, user, start_time):
    _local_start_time = time.time()

    # load user preferences,
    user_prefs = load_user_data(user.username)
    print(">>>>>User preferences: ", type(user_prefs), user_prefs)
    rec = Search_Recipes_filtering(ingredients, user_prefs, query_ranked=True, recipe_range=(0, 3))
    # Load everything without user preferences
    # rec = Search_Recipes(ingredients, query_ranked=True, recipe_range=(0, 3))
    if rec:
        first_recipe_rank = sorted(rec.keys())[0]  # Get the rank of the first recipe
        first_recipe = rec[first_recipe_rank]
        other_recipes_summary = await generate_other_recipes_summary(rec)

        if (verbose):
            message_text = (
                f"Title: \n{first_recipe.title}\n"
                f"\nIngredients: \n{first_recipe.ingredients}\n"
                f"\nInstructions: \n{first_recipe.instructions}\n"
                f"\nTook {round(time.time() - _local_start_time, 2)} seconds"
            )
            await send_split_message(update, message_text)

        dalle_time = time.time()
        dallae_img_path = await generate_and_send_dalle_image(first_recipe, update)
        if (verbose):
            await update.message.reply_text(
                f"Creating Dall-e image\n"
                f"Took {round(time.time() - dalle_time, 2)} seconds\n\n"
            )

        pdf_path = save_pdf_style2(user, ingredients, first_recipe, dallae_img_path, other_recipes_summary)
        await send_final_responses(update, pdf_path, start_time, first_recipe.title)
    else:
        await update.message.reply_text("No recipes found.")


async def generate_and_send_dalle_image(first_recipe, update):

    _img_path = "resources/placeholder_recipe.png" # if not first_recipe.picture_link else first_recipe.picture_link  Imagelink is to localdatabase ofscrapping, not here
    if (generate_photo):
        try:
            _img_path = get_dalle_image(first_recipe.title, first_recipe.ingredients)
            # await update.message.reply_photo(photo=open(_img_path, 'rb'), caption=f"Recipe: {first_recipe.title}") # Display generated photo
        except Exception as e:
            print("Error generating Image with Dall-e:", e)

    return _img_path


async def generate_other_recipes_summary(rec):
    other_recipes_summary = "Other Recipes:\n"
    for rank in sorted(rec.keys())[1:]:
        current_recipe = rec[rank]
        other_recipes_summary += f"Recipe {rank}: {current_recipe.title}\n"
    return other_recipes_summary


async def send_final_responses(update, pdf_path, start_time, title=""):
    # if (verbose):
    #     await update.message.reply_text(f"In total Took {round(time.time() - start_time, 2)} seconds\n\n\n")
    await update.message.reply_text(f"Your recipe was ready in only {round(time.time() - start_time, 2)} seconds\n\n\n")

    if pdf_path != None:
        with open(pdf_path, 'rb') as pdf_file:
            await update.message.reply_document(pdf_file)
    else:
        await update.message.reply_text("Sorry, I couldn't deliver your PDF")

    await update.message.reply_text("Bon Apetit!!!")

    # like or not?
    keyboard = [
        [InlineKeyboardButton("👍 Like", callback_data=f'like::{title}'),
         InlineKeyboardButton("👎 Dislike", callback_data=f'dislike::{title}')]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text('Do you like this recipe?', reply_markup=reply_markup)


async def button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    username = update.effective_user.username

    # This makes the bot edit the message with the button that was clicked to acknowledge the action.
    await query.answer()
    print("You pressed!!!", query.data)
    recipe_title = query.data.split('::')[-1].strip()   # racipes with dashes will be saved wrong
    if query.data.startswith("like"):
        # save to ./user_data/user_prefs.json
        message = f"We are glad you like {recipe_title}"
        save_to_list = 'whitelist'
    else:
        message = f"{recipe_title} Won't be recommended anymore!"
        save_to_list = 'blacklist'

    # Load current preferences
    with open(USER_PREFS_JSON_PATH, 'r') as file:
        data = json.load(file)

    # Check if the user already exists in the JSON data
    if username in data:
        # If the user exists, append the title to their whitelist, avoiding duplicates
        if save_to_list in data[username]:
            if recipe_title not in data[username][save_to_list]:
                data[username][save_to_list].append(recipe_title)
        else:
            # If there's no whitelist for the user, create one
            data[username][save_to_list] = [recipe_title]
    else:
        # If the user does not exist, create a new entry
        data[username] = {save_to_list: [recipe_title]}

    # Write the updated preferences back to the file
    with open(USER_PREFS_JSON_PATH, 'w') as file:
        json.dump(data, file, indent=4)

    await query.edit_message_text(text=message)


async def configure_bot_by_text(full_text, update):

    full_text = full_text.replace("configure ", "").replace("/" , "")
    if ("help" in full_text or "about" in full_text or "" == full_text):
        await update.message.reply_text(
            f"---Help---\n"
            f"Format for configuration should be separated by comma:\n"
            f"configure: verbose=true, debug-false, ... \n\n"
            f"Configuration options are:\n"
            f"- verbose=boolean\n"
            f"- debug=boolean\n"
            f"- generate_photo=boolean\n"

            f" verbose True"
        )
    else:
        commands_string = full_text.replace("configure ", "").replace("help ", "").replace("about ", "").replace(":", "").replace("-","=").replace(" ", "=") #it was empty not =
        if (commands_string != ""):
            config_items = commands_string.split(", ")
            config_dict = {}
            for item in config_items:
                key, value_str = item.split("=")
                value = True if value_str == "true" else False
                config_dict[key] = value
            # Asign the variables to the globals
            for key, value in config_dict.items():
                if key in globals():
                    globals()[key] = value

        await update.message.reply_text('All Done! Configurations should be ready for your bot! Send your ingredients now!')
    # await update.message.reply_text('All Done! Send your ingredients now!', reply_markup= InlineKeyboardMarkup( [[InlineKeyboardButton( text = 'Help', callback_data='help')]]))


# ----------------------------- MAIN COMMANDS ---------------------------------------------

# Main photo handling function with helper functions integration
async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    start_time = time.time()
    photo_file = await update.message.photo[-1].get_file()
    user = update.message.from_user

    await ensure_user_folder_exists(user)

    file_path = f"./user_data/{user.first_name}/{photo_file.file_id}.jpg"
    await photo_file.download_to_drive(file_path)

    if (verbose):
        await update.message.reply_text('Photo received! I\'m counting potatoes.....')
    else:
        await update.message.reply_text('Photo received!\nIm reading 9582 cooking books to find your best recipe!.....')

    # Process for extracting ingredients from photo
    ingredients = get_ingredients(file_path, detailed=False, debug=False, _CONFIDENCE_THRESH=0.75)
    ingredients = sort_ingredients(ingredients)

    if (verbose):
        await update.message.reply_text(
            f"Ingredients detected: \n\n{list(ingredients)}\n\n"
            f"Took {round(time.time() - start_time, 2)} seconds\n\n"
            f"Im reading 9582 cooking books to find your best recipe!\n"
        )

    await process_recipes_and_respond(ingredients, update, user, start_time)


async def text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    start_time = time.time()
    user = update.message.from_user
    full_text = update.message.text

    full_text = full_text.lower()  # should I respect capitals and so?

    if (full_text.startswith("configure")):
        await configure_bot_by_text(full_text, update)
        return

    # Process for extracting ingredients from text
    ingredients = full_text.split(',')
    ingredients = sorted(ingredients)

    await update.message.reply_text('Got it!\nIm reading 9582 cooking books to find your best recipe!.....')

    await process_recipes_and_respond(ingredients, update, user, start_time)


##### AUDIO ---------------------------------

async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """VOICE is a voicenote, using the microphone directly."""
    audio_file = await update.message.voice.get_file()
    await process_audio_message(audio_file, update, "voice")


async def audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """AUDIO will receive an audio file from the device."""
    audio_file = await update.message.audio.get_file()
    await process_audio_message(audio_file, update, "audio")


async def process_audio_message(audio_file, update, audio_source) -> int:
    """Process an audio message from the user."""
    print(f"Running from {audio_source}")
    start_time = time.time()

    await update.message.reply_text("I am listening to it, give me a second....")
    user = update.message.from_user

    randvar = random.randint(5000, 9999)
    audio_filename = f"user_voice-{randvar}.ogg"
    audio_path = f"./{user_data}/{user.first_name}/{audio_filename}"

    if not os.path.exists(os.path.join('./', (user_data), user.first_name)):
        print(f"Making a folder for {user_data}/{user.first_name}")
        os.makedirs(os.path.join('./', (user_data), user.first_name))

    await audio_file.download_to_drive(audio_path)
    logger.info("Audio of %s: %s", user.first_name, audio_filename)

    trans = get_transcript(audio_path)
    if verbose:
        await update.message.reply_text(f"You said:\n{trans}")

    ingredients = trans.split(',')
    ingredients = sorted(ingredients)
    await update.message.reply_text('Got it!\nIm reading 9582 cooking books to find your best recipe!.....')

    await process_recipes_and_respond(ingredients, update, user, start_time)


#

##############
### BOT COMMANDS
##############

# ---- USER PREFS ----


async def show_user_prefs(update: Update, context: CallbackContext) -> int:
    username = update.effective_user.username
    if os.path.exists(USER_PREFS_JSON_PATH):
        with open(USER_PREFS_JSON_PATH, 'r+') as file:
            data = json.load(file)
            if username not in data:
                await update.message.reply_text(f"There is no data for the user:{username}")
                return ConversationHandler.END
            await update.message.reply_text(f"Preferences for {username}")

            await update.message.reply_text(f"Allergies:\n{', '.join(data[username]['allergies'])}")
            blacklist='\n* '.join(data[username]['blacklist'])
            await update.message.reply_text(f"Disliked Recipes:\n* {blacklist}")
            whitelist='\n* '.join(data[username]['whitelist'])
            await update.message.reply_text(f"Liked Recipes:\n* {whitelist}")
            # await update.message.reply_text(f"Diet:\n{', '.join(data[username]['diet'])}")

    # RETURN TO FUNCTION READING ALLERGIES from user
    return ConversationHandler.END


async def save_allergies(update: Update, context: CallbackContext) -> int:
    text = update.message.text
    text=text.replace(":", "").replace("/allergies","")
    if(text != ""):
        allergies = [allergy.strip() for allergy in text.split(',')]

        username = update.effective_user.username
        update_user_allergies(username, allergies)

        await update.message.reply_text(f">Your allergies have been updated to: {', '.join(allergies)}")
    else:
        await update.message.reply_text(f"Send your allergies in the format: \n/allergies nuts, soy, strawberries, etc")

    # RETURN TO FUNCTION READING ALLERGIES from user
    return ConversationHandler.END


def update_user_allergies(username: str, new_allergies: list):
    if os.path.exists(USER_PREFS_JSON_PATH):
        with open(USER_PREFS_JSON_PATH, 'r+') as file:
            data = json.load(file)
            if username not in data:
                data[username] = {'allergies': []}
            elif 'allergies' not in data[username]:
                data[username]['allergies'] = []
            current_allergies = set(data[username]['allergies'])
            updated_allergies = list(current_allergies.union(set(new_allergies)))
            data[username]['allergies'] = updated_allergies
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()  # Remove remaining parts of old data
    else:
        with open(USER_PREFS_JSON_PATH, 'w') as file:
            json.dump({username: {'allergies': new_allergies}}, file, indent=4)

async def configure_bot_command(update: Update, context: CallbackContext):
    _text = update.message.text
    _text = _text.replace("/configure","")
    await configure_bot_by_text(_text, update)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and asks the user to send a voicenote."""

    await update.message.reply_text(
        f"Hello! This is a {BOTNAME}. \nYour Diet Coach.\n"
        f"I will suggest recipes based on the food on your fridge or photos\n"
        f"Send me a photo of the ingredients \n"
        f"or voicenote listing ingredients.\n"
        f"RECIPIX WORKS BETTER with a photo of MULTIPLE ingredients.\n"
        "Start sending a Photo, Voice-Note, or a Text"
    )
    # return AUDIO


async def details(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and asks the user to send a voicenote."""
    config_model_sic = get_model_config()
    await update.message.reply_text(
        f"Hello! This is a {BOTNAME}. \nYour Diet Coach.\n"
        f"This are all the details of the bot:\n"
        f""
        f"{config_model_sic}"
        f""
    )
    # return AUDIO


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    await update.message.reply_text(
        f"{BOTNAME} wishes you well, please visit your doctor regularly.", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


# Create the error handler function
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error."""
    logging.error(msg="Exception occurred", exc_info=context.error)
    # await update.message.reply_text(
    #     f"{BOTNAME} got confused!!\nThere was an error. Please try again.", reply_markup=ReplyKeyboardRemove()
    # )
    if update.effective_message:
        await update.effective_message.reply_text(f"{BOTNAME} got confused!!\nThere was an error. Please try again.",
                                                  reply_markup=ReplyKeyboardRemove())
    else:
        # Log the error or notify the developer, as the bot can't send a message directly to the user
        print("An error occurred, but no message context is available to reply to.")


def main() -> None:
    """Start the amazing CareNavi bot."""
    application = Application.builder().token(TOKEN).build()
    print(f">>>>> {BOTNAME} IS ALIVE <<<<<<<<<<<<<<<<")

    # conv_handler = ConversationHandler(
    #     entry_points=[CommandHandler("start", start)],
    #     states={
    #     AUDIO: [
    #     MessageHandler(filters.VOICE, voice),
    #     MessageHandler(filters.AUDIO, audio),
    #     MessageHandler(, text)
    #                 ], 
    #     },
    #     fallbacks=[CommandHandler("cancel", cancel)],
    # )

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start),
                      CommandHandler("allergies", save_allergies),
                      CommandHandler("configure", configure_bot_command),
                      CommandHandler("user", show_user_prefs),

                      ],
        states={
            #States are run in order -> setup user?->diet->allergies->run
            # ALLERGIES: [
            #     # MessageHandler(filters.TEXT, save_allergies),
            # ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    # Add the error handler
    application.add_error_handler(error_handler)
    application.add_handler(conv_handler)

    application.add_handler(MessageHandler(filters.VOICE, voice))
    application.add_handler(MessageHandler(filters.TEXT, text))
    application.add_handler(MessageHandler(filters.AUDIO, audio))
    application.add_handler(MessageHandler(filters.PHOTO, photo))

    application.add_handler(CallbackQueryHandler(button))

    application.add_handler(conv_handler)
    application.run_polling()


if __name__ == "__main__":
    main()

# Made with <3
# by Ivan Zepeda
# github@ijzepeda-LC

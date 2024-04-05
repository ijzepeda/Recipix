from openai import OpenAI
import requests
import toml

def get_dalle_image(recipe_title, ingredientes):
    api_key=toml.load('./secrets.toml')['OPENAI_API_KEY']
    client = OpenAI(api_key=api_key)
    _prompt=f"Give me an realistic photo of food based on the recipe '{recipe_title}' using the following ingredients and avoid generating text:{ingredientes}"
    print("Dalle Prompt: "+_prompt)
    response = client.images.generate(
      model="dall-e-2",
      prompt=_prompt,
      size="512x512",
      quality="standard",
      n=1,
    )

    image_url = response.data[0].url

    image_response = requests.get(image_url)
    # Check if the request was successful
    if image_response.status_code == 200:
        image_path= "./user_data/suggested_recipe.png"
        # Open a file in binary write mode and save the image content
        with open(image_path, "wb") as file:
            file.write(image_response.content)
        print("Image downloaded and saved as 'downloaded_image.png'.")
        return image_path

    else:
        print("Failed to download the image.")
        return None



# Made with <3
# by Ivan Zepeda
# github@ijzepeda-LC
from fpdf import FPDF
import os
import datetime
import random

t = datetime.datetime.today()
D = t.strftime('%B %d, %y')
D_file = t.strftime('%d-%m-%y_%H%M%S')
PDF_FOLDER = "./user_data/pdfs"
LOGO_PATH = "./resources/recipix_logo.png" #es desde donde se ejecuta, no del origen de la libreria
LOGO_TRANSPARENT_PATH ='./resources/recipix_logo_transparent.png'

# if folder doesn't exist create it
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)


def is_not_valid_data(dictionary):
    for value in dictionary.values():
        if value is not None and value:
            return False
    return True



def split_long_text(pdf, long_text, page_width=210, cell_height=10, font_size=24, separators=[] ):
    if(page_width>100):
        starting_point = page_width
    else:
        page_width=150
        starting_point =20
    if(type(long_text) == list):
        long_text = ", ".join(long_text)

    for separator in separators:
        long_text = long_text.replace(separator, '\n* ') #// mas de 8 items excede

    lines = long_text.split("\n")
    # cell_height = 5 if font_size <= 10 else cell_height  # TODO: modified, not here
    split_lines = []
    for line in lines:
        while len(line) > 0:
            max_chars_per_line = 15 + abs(70 - (font_size * 3))  # 17
            max_chars_per_line =  abs(page_width - (font_size * 3))  # 17
            # If the line is short enough, add it as is
            if len(line) <= max_chars_per_line:
                split_lines.append(line)
                break
            # If the line is too long, split at the nearest space before the limit
            split_point = line.rfind(' ', 0, max_chars_per_line)
            if split_point == -1:  # No space found, force split at max_chars_per_line
                split_point = max_chars_per_line
            # Add the split line to the list
            split_lines.append(line[:split_point].strip())
            # Remove the processed part from the line
            line = line[split_point:].lstrip()

    # Add each chunk into the PDF as a new cell
    for split_line in split_lines:
        pdf.set_x(starting_point / 2)
        pdf.cell(0, cell_height, str(split_line).strip(), 0, 1, 'L')
    return len(split_lines)

def save_pdf_style2(user, ingredients, recipe, dallae_img_path, other_recipes):
    # Create instance of FPDF class
    pdf = FPDF(orientation='P', unit='mm', format='A4')

    # Add a page
    pdf.add_page()

    page_width = pdf.w
    page_height = pdf.h

    # ------ colors -------------------
    color_purple = (152, 4, 176)  # 255.0.0
    color_red = (220, 50, 50)
    color_blue = (41, 26, 99)
    color_beige = (255, 226, 182)
    color_white = (255, 255, 255)
    color_black = (0, 0, 0)
    color_cream = (245, 245, 220)
    color_grey = (217, 217, 217)
    # (255, 253, 208)  # RGB for Cream
    accent_color = color_purple
    second_color = color_beige
    title_color = color_blue
    banner_text_color = color_beige
    banner_color = color_red
    background_color = color_beige

    # --- PAge setup----------------------
    pdf.set_fill_color(background_color[0], background_color[1], background_color[2])
    pdf.rect(0, 0, pdf.w, pdf.h, 'F')
    border_width = 1  # Width of the border
    pdf.set_line_width(border_width)
    pdf.set_draw_color(accent_color[0], accent_color[1], accent_color[2])
    pdf.rect(5, 5, pdf.w - 10, pdf.h - 10)  # Adjust the dimensions as needed

    # HEADER-----------------------------
    # Set title
    pdf.set_y(11)
    pdf.set_font("Arial", 'B', 30)
    pdf.set_text_color(title_color[0], title_color[1], title_color[2])
    pdf.cell(200, 10, "Recipix", 0, 1, 'C')
    pdf.set_font("Arial", 'B', 15)
    pdf.cell(200, 10, "Your food coach", 0, 1, 'C')

    # Draw a colored line
    pdf.set_draw_color(accent_color[0], accent_color[1], accent_color[2])
    pdf.set_line_width(1)
    pdf.line(10, 35, 200, 35)

    # Add a logo
    pdf.image(LOGO_PATH, x=10, y=3, w=35)

    pdf.image(LOGO_TRANSPARENT_PATH, x=10, y=80, w=190)
    # ------------------------------------------------------
    # INTENTO DE FOTOYTITULO
    # Set title
    pdf.set_y(40)
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(title_color[0], title_color[1], title_color[2])
    pdf.set_x(page_width / 2)
    split_long_text(pdf, recipe.title, page_width=page_width, font_size=24,cell_height=10)  # TODO <<<<<<<<<<<<<<<<<<<

    pdf.set_font("Arial", 'B', 16)
    pdf.set_x(page_width / 2)
    pdf.cell(50, 10, f"Tailored for {user['first_name']}", 0, 1, 'L')
    pdf.set_x(page_width / 2)
    # pdf.ln(2)
    # pdf.set_font("Arial",'', 12)
    # ingredienteses=f"Found Ingredients:\n{ingredients}"
    # split_long_text(pdf, ingredienteses, page_width=page_width , font_size=12, separators=[',',';']) #TODO <<<<<<<<<<<<<<<<<<< depende del size del font

    # DALL-e image food image
    pdf.image(dallae_img_path, x=15, y=40, w=page_width * 0.4)
    pdf.set_draw_color(accent_color[0], accent_color[1], accent_color[2])
    pdf.rect(15, 40, page_width * 0.4, page_width * 0.4)

    # --------------------CONTENT-----------------------

    # Section title: Ingredients
    # pdf.ln(45)
    pdf.ln(2)
    pdf.set_x(page_width / 2)

    pdf.set_font("Arial", 'B', 20)
    pdf.set_fill_color(banner_color[0], banner_color[1], banner_color[2])  # Red fill color
    pdf.set_text_color(banner_text_color[0], banner_text_color[1], banner_text_color[2])
    pdf.cell(70, 10, " INGREDIENTS", 0, 1, 'L', 1)
    pdf.ln(3)

    # Ingredients list
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(0, 0, 0)
    # pdf.multi_cell(0, 10, recipe['ingredients'])
    # TODO <depende el formato que venga comma is reserved for steps
    count_ings= split_long_text(pdf, recipe.ingredients , page_width=page_width,cell_height=5, font_size=10, separators=[ ';'])
    #--------------------------------------------------

    pdf.set_y((page_height / 2) - 15 + count_ings)
    # Section title: Instructions
    pdf.set_font("Arial", 'B', 20)
    pdf.set_fill_color(banner_color[0], banner_color[1], banner_color[2])
    pdf.set_text_color(banner_text_color[0], banner_text_color[1], banner_text_color[2])
    pdf.cell(0, 10, " INSTRUCTIONS", 0, 1, 'L', 1)

    # Instructions list
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, recipe.instructions )
    #--------------------------------------------------
    pdf.ln(2)

    # Section title: Other Recipes
    pdf.set_font("Arial", 'B', 18)
    pdf.set_fill_color(banner_color[0], banner_color[1], banner_color[2])
    pdf.set_text_color(banner_text_color[0], banner_text_color[1], banner_text_color[2])
    pdf.cell(0, 10, " OTHER RECIPES:", 0, 1, 'L', 1)

    # Other recipes list
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 5, other_recipes)
    #--------------------------------------------------

    pdf.ln(2)
    pdf.set_font("Arial", 'UI', 10)
    pdf.cell(0, 10, f"Recipe based on the ingredients:")
    pdf.ln()
    split_long_text(pdf, ingredients, cell_height=3, page_width=20, font_size=10 ) #se puede pasar a una segunda pagina

    # Output the PDF
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)

    randvar = random.randint(50000, 100000)
    pdf_path = os.path.join(PDF_FOLDER, f"Recipe-{recipe.title}-for-{user['first_name']}_{randvar}.pdf")
    pdf.output(pdf_path, 'F')  # .encode('latin-1','ignore').decode('latin-1')
    print("PDF SAVED AT", pdf_path)
    return pdf_path






# Made with <3
# by Ivan Zepeda
# github@ijzepeda-LC
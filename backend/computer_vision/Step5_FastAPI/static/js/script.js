document.addEventListener("DOMContentLoaded", function() {
const uploadForms = document.querySelectorAll('.uploadForm');
uploadForms.forEach(form => {
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // if (this.getAttribute('data-submitting') === 'true') return; // Prevent double submission
        // this.setAttribute('data-submitting', 'true'); // Mark form as submitting

        document.getElementById('spinnerOverlay').style.display = 'flex';

        const formData = new FormData(this);
        const actionUrl = this.getAttribute('action');

        try {
            const response = await fetch(actionUrl, {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) throw new Error('Network response was not ok.');
            const data = await response.json();

            if(data && data.JSON) { // Ensure this matches your actual response structure
                displayRecipes(data.JSON);
            } else {
                // Handle unexpected structure or error
                console.error('Invalid data structure:', data);
            }
        } catch (error) {
            console.error('Error:', error);
        } finally {
            document.getElementById('spinnerOverlay').style.display = 'none';
                            // this.removeAttribute('data-submitting'); // Reset submitting state

        }
    });
});
});


// segunda funcion de mostrar recipes
function displayRecipes(data) {
    console.log("displaying recipes")
    document.getElementById('recipesSection').style.display = 'block';
    if (data.top_recipe) {
        console.log("top recipe found on index html")
        document.getElementById('title').textContent = data.top_recipe.title;
        if (data.image_url) {
            const img = document.getElementById('recipeImage');
            img.src = data.image_url;
            img.style.display = 'block';
            img.style.marginLeft = 'auto';
            img.style.marginRight = 'auto';
            img.style.width='50%';
            img.tabIndex=0;
            img.focus();
        }
        document.getElementById('ingredients').textContent = `${data.ingredients}`;
        document.getElementById('instructions').textContent = `${data.instructions}`;

    }

    if (data.other_recipes) {
        console.log("other_recipes found on index html")
        // document.getElementById('other_recipees_title').visibility='visible'

        const otherRecipesList = document.getElementById('otherRecipes');
        otherRecipesList.innerHTML = ''; // Clear existing list items
        data.other_recipes.forEach(recipe => {
            const li = document.createElement('li');
            li.textContent = recipe.title;
            otherRecipesList.appendChild(li);
        });
    }

    if (data.other_recipes) {
        console.log("other_recipes found on index html");

        const otherRecipesList = document.getElementById('otherRecipes');
        otherRecipesList.innerHTML = ''; // Clear existing list items

        data.other_recipes.forEach((recipe, index) => {
            const li = document.createElement('li');
            const title = document.createElement('span');
            title.textContent = recipe.title;
            title.style.cursor = 'pointer';
            title.style.color = '#007bff'; // Style as you see fit
            title.onclick = () => toggleDetails(index); // Toggle details view on click

            const detailsDiv = document.createElement('div');
            detailsDiv.id = `details-${index}`;
            detailsDiv.style.display = 'none'; // Initially hide the details
            detailsDiv.innerHTML = `
                <p><strong>Ingredients:</strong> ${recipe.ingredients}</p>
                <p><strong>Instructions:</strong> ${recipe.instructions}</p>
            `;

            li.appendChild(title);
            li.appendChild(detailsDiv);
            otherRecipesList.appendChild(li);
        });
    }

    // Function to toggle the display of recipe details
    function toggleDetails(index) {
        const detailsDiv = document.getElementById(`details-${index}`);
        detailsDiv.style.display = detailsDiv.style.display === 'none' ? 'block' : 'none';
    }





    if(data.message){
      document.getElementById('message').textContent = data.message;

    }

    document.getElementById('other_recipees_title').style.display = 'block';
    document.getElementById('recipesSection').style.display = 'block';


}

// Async function to display o the go -------------------------------
//
//     async function fetchDataAndUpdateUI(actionUrl, formData) {
//     // Initial Fetch to Get Data
//     try {
//         const response = await fetch(actionUrl, {
//             method: 'POST',
//             body: formData,
//         });
//         if (!response.ok) throw new Error('Network response was not ok.');
//         const data = await response.json();
//
//         // Assuming the data structure is as follows:
//         // data = { ingredients: [], instructions: "", image_url: "" }
//
//         // Update Ingredients
//         if(data.ingredients) {
//
//             document.getElementById('ingredients').textContent = `>Ingredients:\n ${data.ingredients}`;
//         }
//
//         // Update Instructions
//         if(data.instructions) {
//                   document.getElementById('instructions').textContent = `Instructions:\n ${data.instructions}`;
//
//     }
//
//         // Update Image
//         if(data.image_url) {
//             const img = document.getElementById('recipeImage');
//             img.src = data.image_url;
//             img.style.display = 'block';
//         }
//
//     } catch (error) {
//         console.error('Error:', error);
//     } finally {
//         document.getElementById('spinnerOverlay').style.display = 'none';
//     }
//     }
//
//     document.addEventListener("DOMContentLoaded", function() {
//     document.querySelectorAll('.uploadForm').forEach(form => {
//         form.addEventListener('submit', async function(e) {
//             e.preventDefault();
//             document.getElementById('spinnerOverlay').style.display = 'flex';
//
//             const formData = new FormData(this);
//             const actionUrl = this.getAttribute('action');
//
//             // Call the function to fetch data and update UI
//             fetchDataAndUpdateUI(actionUrl, formData);
//         });
//     });
// });





//
// <!--# Made with <3-->
// <!--# by Ivan Zepeda-->
// <!--# github@ijzepeda-LC-->
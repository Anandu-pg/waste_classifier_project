from cProfile import Profile
import os
import numpy as np
from PIL import Image, ImageOps
from django.shortcuts import render, redirect
from tensorflow.keras.models import load_model
from django.core.files.storage import FileSystemStorage # type: ignore
from dotenv import load_dotenv
import google.generativeai as gen_ai
from django.contrib import messages # type: ignore
from django.contrib.auth.models import User # type: ignore
from django.contrib.auth import authenticate, login, logout # type: ignore
from django.http import JsonResponse

from classifier.models import ContactMessage


# Load environment variables
load_dotenv()

# Load your model and labels
model = load_model("models/model.h5", compile=False)
class_names = open("models/labels.txt", "r").readlines()


# Load the labels, removing index numbers
class_names = [line.split(maxsplit=1)[1].strip() for line in open("models/labels.txt", "r").readlines()]


# Function to classify waste
def classify_waste(img):
    np.set_printoptions(suppress=True)

    # Prepare the image for classification
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict with the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]  # Only the class name, no index
    confidence_score = prediction[0][index] * 100  # Convert to percentage
    confidence_score = round(confidence_score, 2)  # Round to 2 decimal places

    return class_name, confidence_score



# Function to generate carbon footprint info
def generate_carbon_footprint_info(label):
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("API key not found. Please ensure the GOOGLE_API_KEY is set in your .env file.")

    # Set up Google Gemini-Pro AI model
    gen_ai.configure(api_key=GEMINI_API_KEY)
    model = gen_ai.GenerativeModel('gemini-pro')

    # Define the prompt to ask about carbon emission
    prompt = f"What is the approximate carbon emission or carbon footprint generated from {label}? Elaborate in 100 words."

    gemini_response = model.generate_content(contents=prompt)
    response_text = gemini_response.text
    word_limit = 100
    truncated_response = ' '.join(response_text.split()[:word_limit])

    return truncated_response

# Function to generate how to recycle or reuse
def reuse(label):
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("API key not found. Please ensure the GOOGLE_API_KEY is set in your .env file.")

    # Set up Google Gemini-Pro AI model
    gen_ai.configure(api_key=GEMINI_API_KEY)
    model = gen_ai.GenerativeModel('gemini-pro')

    # Define the prompt to ask about recycling or reusing
    prompt = f"How to reuse {label}? Elaborate in a paragraph with 100 words."

    gemini_response = model.generate_content(contents=prompt)
    response_text = gemini_response.text
    word_limit = 100
    truncated_response = ' '.join(response_text.split()[:word_limit])

    return truncated_response

# Function to generate how to compost
def compost(label):
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("API key not found. Please ensure the GOOGLE_API_KEY is set in your .env file.")

    # Set up Google Gemini-Pro AI model
    gen_ai.configure(api_key=GEMINI_API_KEY)
    model = gen_ai.GenerativeModel('gemini-pro')

    # Define the prompt to ask about composting
    prompt = f"How to compost {label} that doesn't affect nature badly? Elaborate in a paragraph with 100 words."

    gemini_response = model.generate_content(contents=prompt)
    response_text = gemini_response.text
    word_limit = 100
    truncated_response = ' '.join(response_text.split()[:word_limit])

    return truncated_response


# The view to handle image upload and classification
def classify(request):
    if request.method == 'POST' and request.FILES['image']:
        img = request.FILES['image']
        
        # Handle image file storage
        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        uploaded_file_url = fs.url(filename)
        
        # Classify the waste
        image_file = Image.open(img)
        label, confidence_score = classify_waste(image_file)

        # Get AI response for carbon footprint
        response1 = generate_carbon_footprint_info(label)

       # Get AI response for how to reuse
        response2 = reuse(label)

       # Get AI response for how to compoost
        response3 = compost(label)


        # Return result to template
        context = {
            'image_url': uploaded_file_url,
            'label': label,
            'confidence_score': confidence_score,
            'carbon_footprint': response1,
            'reuse' : response2,
            'compost' : response3,
        }
        
        return render(request, 'result.html', context)

    return render(request, 'upload.html')

def agenda(request):
    return render(request, 'agenda.html')

def about(request):
    return render(request, 'about.html')


def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')

        # Save the form data into the database
        ContactMessage.objects.create(name=name, email=email, message=message)
        
        # Show success message
        messages.success(request, "Your message has been sent successfully!")
        return redirect('contact')  # Redirect back to the contact page

    return render(request, 'contact.html')

def intro(request):
    # if request.user.is_authenticated:
    #     return redirect('home')  # Redirect to home if logged in
    return render(request, 'intro.html')

    

def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("home")  # Redirect to home page after login
        else:
            messages.error(request, "Invalid username or password")
    
    return render(request, "login.html")

def register_view(request):
    if request.method == "POST":
        # Use get() to safely retrieve form data
        full_name = request.POST.get("full_name")
        email = request.POST.get("email")
        username = request.POST.get("username")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")
        phone = request.POST.get("phone")
        address = request.POST.get("address")

        # Check for empty fields
        if not all([full_name, email, username, password1, password2, phone, address]):
            messages.error(request, "All fields are required.")
            return render(request, "register.html", {
                "full_name": full_name,
                "email": email,
                "username": username,
                "phone": phone,
                "address": address
            })

        # Password match check
        if password1 != password2:
            messages.error(request, "Passwords do not match")
        
        # Username and email availability check
        elif User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken")
        
        elif User.objects.filter(email=email).exists():
            messages.error(request, "Email is already registered")
        
        else:
            # Create user if no errors
            user = User.objects.create_user(username=username, email=email, password=password1)
            user.save()

            # Check if the user already has a profile, if not, create one
            if not hasattr(user, 'profile'):
                Profile.objects.create(user=user, full_name=full_name, phone=phone, address=address)

            messages.success(request, "Account created successfully! Please log in.")
            return redirect("login")

    return render(request, "register.html")


def chat_bot(request):  # This will handle the initial page load
    return render(request, "chat.html")  # Just render the template


def get_bot_response(request):  # New view for AJAX calls
    if request.method == "POST":
        user_input = request.POST.get("query", "").strip() # Get query parameter
        if not user_input:
            return JsonResponse({"response": "Please enter a message."})

        GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
        if not GEMINI_API_KEY:
            return JsonResponse({"response": "API key is missing."})

        gen_ai.configure(api_key=GEMINI_API_KEY)
        model = gen_ai.GenerativeModel('gemini-pro')

        try:
            prompt = f"You are a waste management assistant. Answer the following question only in relation to waste management: {user_input}"
            gemini_response = model.generate_content(contents=prompt)
            bot_response = gemini_response.text if gemini_response.text else "I'm not sure about that. Try asking something else related to waste management."
            return JsonResponse({"response": bot_response})  # Return JSON

        except Exception as e:
            return JsonResponse({"response": f"An error occurred: {e}"})
    return JsonResponse({"response": "Invalid request."}) # Handle GET requests


def logout_view(request):
    logout(request)
    return redirect("login")

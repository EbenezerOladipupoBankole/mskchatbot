from dotenv import load_dotenv
import os
import secrets
from datetime import timedelta
import time
import json
import logging
import boto3
from botocore.exceptions import ClientError
from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from flask_cors import CORS
from functools import wraps
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.wrappers import Request, Response
from io import BytesIO
import base64
from werkzeug.datastructures import Headers
import os
print("Current working directory:", os.getcwd())
# print("Template folder path:", app.template_folder)

# --- Load environment variables at the very top ---
load_dotenv()

# Now get the API key before importing genai
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment")

# Configure Gemini API
import google.generativeai as genai
genai.configure(api_key=api_key)

# --- Continue with the rest of your app setup ---

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# --- Session Cookie Settings for OAuth reliability ---
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True only if using HTTPS

# --- Configuration from Environment Variables ---
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))
if app.secret_key == 'your_flask_secret_key_here':
    logging.warning("FLASK_SECRET_KEY is not set or is default. Please set a strong secret key in your environment.")
app.permanent_session_lifetime = timedelta(days=30)

AWS_REGION = os.getenv('AWS_REGION', 'eu-north-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'musicchatbot-assets')
USERS_TABLE_NAME = os.getenv('USERS_TABLE_NAME', 'MusicChatbotUsers')
CHAT_HISTORY_TABLE_NAME = os.getenv('CHAT_HISTORY_TABLE_NAME', 'MusicChatHistory')
MAX_GUEST_MESSAGES = 40

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log environment variables (excluding sensitive values)
logging.info(f"FLASK_SECRET_KEY loaded: {'*' * len(app.secret_key) if app.secret_key else 'None'}")
logging.info(f"AWS_REGION loaded: {AWS_REGION}")
logging.info(f"S3_BUCKET_NAME loaded: {S3_BUCKET_NAME}")
logging.info(f"USERS_TABLE_NAME loaded: {USERS_TABLE_NAME}")
logging.info(f"CHAT_HISTORY_TABLE_NAME loaded: {CHAT_HISTORY_TABLE_NAME}")
logging.info(f"GOOGLE_API_KEY loaded: {'Set' if api_key else 'Not Set'}")
logging.info(f"MAX_GUEST_MESSAGES set to: {MAX_GUEST_MESSAGES}")

# AWS Clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)

# Tables will be initialized later
users_table = None
chat_history_table = None

def init_db_tables():
    """
    Initializes DynamoDB table objects and creates them if they don't exist.
    """
    global users_table, chat_history_table
    # Initialize Users Table
    try:
        users_table = dynamodb.Table(USERS_TABLE_NAME)
        users_table.load() # Attempt to load table description to check existence
        logging.info(f"DynamoDB Users Table '{USERS_TABLE_NAME}' already exists.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.warning(f"Users table '{USERS_TABLE_NAME}' not found. Creating table...")
            try:
                users_table = dynamodb.create_table(
                    TableName=USERS_TABLE_NAME,
                    KeySchema=[{'AttributeName': 'username', 'KeyType': 'HASH'}],
                    AttributeDefinitions=[{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'user_id', 'AttributeType': 'S'}],
                    ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5},
                    GlobalSecondaryIndexes=[{
                        'IndexName': 'UserIdIndex',
                        'KeySchema': [{'AttributeName': 'user_id', 'KeyType': 'HASH'}],
                        'Projection': {'ProjectionType': 'ALL'},
                        'ProvisionedThroughput': {'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
                    }]
                )
                users_table.wait_until_exists()
                logging.info(f"Users table '{USERS_TABLE_NAME}' created successfully.")
            except ClientError as create_e:
                logging.error(f"Error creating users table '{USERS_TABLE_NAME}': {create_e}")
                users_table = None
        else:
            logging.error(f"Error initializing users table '{USERS_TABLE_NAME}': {e}")
            users_table = None

    # Initialize Chat History Table
    try:
        chat_history_table = dynamodb.Table(CHAT_HISTORY_TABLE_NAME)
        chat_history_table.load() # Attempt to load table description to check existence
        logging.info(f"DynamoDB Chat History Table '{CHAT_HISTORY_TABLE_NAME}' already exists.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.warning(f"Chat history table '{CHAT_HISTORY_TABLE_NAME}' not found. Creating table...")
            try:
                chat_history_table = dynamodb.create_table(
                    TableName=CHAT_HISTORY_TABLE_NAME,
                    KeySchema=[{'AttributeName': 'user_id', 'KeyType': 'HASH'}],
                    AttributeDefinitions=[{'AttributeName': 'user_id', 'AttributeType': 'S'}],
                    ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
                )
                chat_history_table.wait_until_exists()
                logging.info(f"Chat history table '{CHAT_HISTORY_TABLE_NAME}' created successfully.")
            except ClientError as create_e:
                logging.error(f"Error creating chat history table '{CHAT_HISTORY_TABLE_NAME}': {create_e}")
                chat_history_table = None
        else:
            logging.error(f"Error initializing chat history table '{CHAT_HISTORY_TABLE_NAME}': {e}")
            chat_history_table = None

# Initialize tables on app startup
with app.app_context():
    init_db_tables()


def load_user_chat_history(user_id):
    """
    Loads chat history for a given user from DynamoDB.
    Returns the chat history as a list of dictionaries, or an empty list if not found.
    """
    if not chat_history_table:
        logging.error("Chat history table not initialized. Cannot load history.")
        return []
    try:
        response = chat_history_table.get_item(Key={'user_id': user_id})
        item = response.get('Item')
        if item and 'history' in item:
            history = json.loads(item['history'])
            logging.info(f"Successfully loaded chat history for user: {user_id}. History length: {len(history)}")
            return history
        logging.info(f"No chat history found for user: {user_id}")
        return []
    except ClientError as e:
        logging.error(f"DynamoDB error loading chat history for user {user_id}: {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding chat history JSON for user {user_id}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error in load_user_chat_history for user {user_id}: {e}")
        return []

def save_user_chat_history(user_id, history):
    """
    Saves chat history for a given user to DynamoDB.
    History is stored as a JSON string.
    """
    if not chat_history_table:
        logging.error("Chat history table not initialized. Cannot save history.")
        return False
    try:
        chat_history_json = json.dumps(history)
        chat_history_table.put_item(
            Item={
                'user_id': user_id,
                'history': chat_history_json,
                'last_updated': int(time.time()) # Optional: add a timestamp
            }
        )
        logging.info(f"Successfully saved chat history for user: {user_id}. History length: {len(history)}")
        return True
    except ClientError as e:
        logging.error(f"DynamoDB error saving chat history for user {user_id}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error in save_user_chat_history for user {user_id}: {e}")
        return False

# Function to load persona from S3
def load_persona_from_s3():
    try:
        if not S3_BUCKET_NAME:
            logging.warning("S3_BUCKET_NAME is not set. Skipping persona loading from S3.")
            return None
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key='persona.json')
        persona_data = json.loads(response['Body'].read().decode('utf-8'))
        logging.info(f"Successfully loaded persona from S3 bucket: {S3_BUCKET_NAME}/persona.json")
        return persona_data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logging.warning(f"Persona file 'persona.json' not found in bucket '{S3_BUCKET_NAME}'.")
        else:
            logging.error(f"Failed to load persona.json from S3 bucket '{S3_BUCKET_NAME}': {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode persona.json: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading persona from S3: {e}")
        return None

# Load persona on app startup
persona = load_persona_from_s3()
if persona:
    logging.info("Persona loaded successfully.")
else:
    logging.warning("Failed to load persona. Chatbot might not function as expected without a persona.")


# Initialize the Generative Model (using 1.5 Flash as it's efficient for chat)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- Persona instruction moved into the beginning of the history ---
# This serves as a base for the chatbot's identity and behavior
initial_persona_prompt = [
    {
        "role": "user",
        "parts": [{"text": "You are Musik Assitant, a helpful chat assistant created by Ebenezer Musik Assitantal. You will always respond as Musik Assitant and adhere to the persona provided in your training data."}],
    },
    {
        "role": "model",
        "parts": [{"text": "Okay, I understand. I am Musik Assitant, and I will respond as a helpful chat assistant created by Ebenezer Musik Assitantal, adhering to my defined persona."}]
    }
]

# Helper function to get/initialize chat session history for a user
def get_gemini_chat_session_history():
    """
    Retrieves or initializes the chat history for the current user session.
    Includes the initial persona prompt and S3-loaded persona examples.
    """
    user_id = session.get('user_id')

    # Try to load from DynamoDB if user is logged in
    if user_id and session.get('logged_in'): # Only try to load from DB if truly logged in
        user_chat_history = load_user_chat_history(user_id)
        if user_chat_history:
            logging.info(f"Loaded chat session from DB for user: {session.get('username', 'Guest')}")
            return user_chat_history

    # If no history in DB or not logged in, initialize a new history
    new_history = list(initial_persona_prompt) # Start with base instructions

    # Add examples from persona.json if available
    if persona and 'examples' in persona and persona['examples']:
        for example in persona['examples']:
            if 'user' in example and 'bot' in example:
                new_history.append({"role": "user", "parts": [{"text": example['user']}]})
                new_history.append({"role": "model", "parts": [{"text": example['bot']}]})
    
    logging.info(f"Initialized new chat session for user: {session.get('username', 'Anonymous/Guest')}")
    return new_history

# --- Authentication Decorator and Routes with DynamoDB ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            logging.warning("Unauthorized access attempt (not logged in).")
            return jsonify({"error": "Unauthorized. Please log in."}), 401
        if not users_table:
            logging.error("Users table not initialized. Authentication disabled.")
            return jsonify({"error": "Authentication system not configured."}), 500
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/settings')
# @login_required
# def settings():
#     return render_template('settings.html')

# @app.route('/')
# def index():
#     # Render the single index.html file, which now contains all modal HTML
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     if 'user_id' in session and session.get('logged_in'):
#         username = session['username']
#         logging.info(f"User '{username}' accessed /about.")
#     else:
#         logging.info("Guest user accessed /about.")
    
#     return render_template('about.html')


# @app.route('/settings')
# @login_required
# def settings():
#     logging.info(f"User '{session.get('username', 'Anonymous')}' accessed /settings.")
#     return render_template('settings.html')

@app.route('/register', methods=['POST'])
def register():
    if not users_table:
        logging.error("Users table not initialized. Registration disabled.")
        return jsonify({"error": "Registration system not configured."}), 500

    username = request.json.get('username')
    email = request.json.get('email') # Get email from request
    password = request.json.get('password')

    if not username or not email or not password: # Ensure email is also required for registration
        logging.warning("Registration attempt with missing username, email, or password.")
        return jsonify({"error": "Username, email, and password are required"}), 400

    try:
        # Check if username already exists
        response = users_table.get_item(Key={'username': username})
        if 'Item' in response:
            logging.warning(f"Registration attempt for existing username: {username}.")
            return jsonify({"error": "Username already exists"}), 409 # 409 Conflict

        # Hash password
        password_hash = generate_password_hash(password)

        # Generate a unique user_id (different from username, for flexibility if usernames change or for internal use)
        user_id = secrets.token_hex(16)

        users_table.put_item(
            Item={
                'username': username,
                'password_hash': password_hash,
                'user_id': user_id, # Store a unique ID for session tracking
                'email': email # Store the email
            }
        )
        logging.info(f"User '{username}' registered successfully with user_id: {user_id}")
        return jsonify({"message": "Registration successful"}), 201 # 201 Created
    except ClientError as e:
        logging.error(f"DynamoDB error during registration for user '{username}': {e}", exc_info=True)
        return jsonify({"error": "Registration failed due to server error."}), 500
    except Exception as e:
        logging.error(f"Error during registration for user '{username}': {e}", exc_info=True)
        return jsonify({"error": "Registration failed due to server error."}), 500


@app.route('/login', methods=['POST'])
def login():
    if not users_table:
        logging.error("Users table not initialized. Login disabled.")
        return jsonify({"error": "Login system not configured."}), 500

    username = request.json.get('username')
    password = request.json.get('password')

    try:
        response = users_table.get_item(Key={'username': username})
        user_data = response.get('Item')

        if user_data and check_password_hash(user_data['password_hash'], password):
            session['logged_in'] = True
            session['username'] = user_data['username']
            session['user_id'] = user_data['user_id'] # Use the user_id stored in DynamoDB
            session.permanent = True

            # Load chat history for the logged-in user into session
            session['user_chat_history'] = get_gemini_chat_session_history() # Calls the new function
            
            logging.info(f"User '{username}' logged in. Chat history initialized/loaded.")
            # Include email in response if it exists in user_data
            return jsonify({"message": "Login successful", "username": user_data['username'], "email": user_data.get('email')}), 200
        else:
            logging.warning(f"Login failed for user '{username}'. Invalid credentials.")
            session.pop('logged_in', None)
            session.pop('username', None)
            session.pop('user_id', None)
            session.pop('user_chat_history', None)
            session.permanent = False
            return jsonify({"error": "Invalid credentials"}), 401
    except ClientError as e:
        logging.error(f"DynamoDB error during login for user '{username}': {e}", exc_info=True)
        return jsonify({"error": "Login failed due to server error."}), 500
    except Exception as e:
        logging.error(f"Error during login for user '{username}': {e}", exc_info=True)
        return jsonify({"error": "Login failed due to server error."}), 500

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    user_id = session.get('user_id')
    if user_id and 'user_chat_history' in session:
        # Save current chat history to DynamoDB before logging out
        save_user_chat_history(user_id, session['user_chat_history'])

    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('user_chat_history', None)
    session.pop('guest_message_count', None) # Clear guest count on logout
    session.permanent = False
    logging.info(f"User '{session.get('username', 'Anonymous')}' logged out. Chat history cleared/saved.")
    return jsonify({"message": "Logged out"}), 200

@app.route('/check_login_status', methods=['GET'])
def check_login_status():
    if 'user_id' in session and session.get('logged_in'):
        user_id = session['user_id']
        username = session['username'] # Get username directly from session

        # Ensure chat history is loaded into session if not already present (e.g., on first page load or session restore)
        if 'user_chat_history' not in session:
            session['user_chat_history'] = get_gemini_chat_session_history()
            logging.info(f"Chat history re-initialized/loaded into session for user {user_id}.")
            
        # Filter out persona prompts from the history sent to the frontend for display
        display_chat_history = []
        for msg in session['user_chat_history']:
            # Get the text content safely, handling both dict {'text': '...'} and raw string formats
            message_text = ""
            if msg['parts'] and isinstance(msg['parts'], list) and len(msg['parts']) > 0:
                first_part = msg['parts'][0]
                if isinstance(first_part, dict) and 'text' in first_part:
                    message_text = first_part['text']
                elif isinstance(first_part, str):
                    message_text = first_part
            
            # Check if the message's text matches either of the initial persona prompts
            is_persona_message = False
            if msg['role'] == 'user' and message_text == initial_persona_prompt[0]['parts'][0]['text']:
                is_persona_message = True
            elif msg['role'] == 'model' and message_text == initial_persona_prompt[1]['parts'][0]['text']:
                is_persona_message = True
            
            if not is_persona_message: # Only add if it's NOT a persona message
                # Append to display_chat_history ensuring the correct format: {'role': ..., 'parts': ['text_content']}
                display_chat_history.append({'role': msg['role'], 'parts': [message_text]})
        
        logging.info(f"Login status check: User '{username}' is logged in. History length for display: {len(display_chat_history)}")
        return jsonify({
            "logged_in": True, 
            "username": username,
            "user_id": user_id, # Frontend might need user_id for client-side distinctions
            "email": session.get('email', None), # Get email from session, if available
            "chat_history": display_chat_history # Send cleaned history for display
        }), 200 # Always return 200 OK if the check itself was successful
    else:
        logging.info("Login status check: User is not logged in.")
        # Return the actual guest message count from the session
        guest_msg_count = session.get('guest_message_count', 0)
        return jsonify({
            "logged_in": False, 
            "guest_message_count": guest_msg_count
        }), 200

# Load hymn book data
with open("hymn_book.json", "r", encoding="utf-8") as f:
    hymn_book = json.load(f)

def get_hymn_lyrics_by_number(number):
    for hymn in hymn_book:
        if str(hymn.get("number")) == str(number):
            return f"{hymn.get('title', '')}\n{hymn.get('lyrics', '')}"
    return None

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_message = request.json.get('message')
    if not user_message:
        logging.warning("Chat attempt with no message provided.")
        return jsonify({"error": "No message provided"}), 400

    # Intercept hymn-related requests (lyrics, info, or just number)
    import re
    hymn_patterns = [
        r"(?i)what are the lyrics to hymn (\\d+)",
        r"(?i)what is hymn (\\d+)",
        r"(?i)show hymn (\\d+)",
        r"(?i)hymn (\\d+)",
        r"(?i)tell me about hymn (\\d+)"
    ]
    for pattern in hymn_patterns:
        match = re.match(pattern, user_message.strip())
        if match:
            hymn_number = match.group(1)
            lyrics = get_hymn_lyrics_by_number(hymn_number)
            if lyrics:
                return jsonify({"response": lyrics})
            else:
                return jsonify({"response": f"Sorry, I couldn't find hymn {hymn_number}."})

    # Guest or logged-in user
    logged_in = session.get('logged_in', False)
    user_id = session.get('user_id')
    username = session.get('username', 'Guest')
    session.permanent = True

    # Initialize or reset chat history if needed
    if 'user_chat_history' not in session:
        session['user_chat_history'] = get_gemini_chat_session_history()
    else:
        # Clean old-format history if present
        if any('content' in entry for entry in session['user_chat_history']):
            session.pop('user_chat_history')
            session['user_chat_history'] = get_gemini_chat_session_history()
            logging.info("Old-format chat history detected and reset.")

    # Guest message limit logic
    if not logged_in:
        guest_message_count = session.get('guest_message_count', 0)
        if guest_message_count >= MAX_GUEST_MESSAGES:
            return jsonify({
                "error": f"You've reached the limit of {MAX_GUEST_MESSAGES} free messages. Please log in.",
                "code": "LIMIT_EXCEEDED"
            }), 403
        session['guest_message_count'] = guest_message_count + 1
        session.modified = True

    # Add new message in Gemini format
    user_entry = {"role": "user", "parts": [{"text": user_message}]}
    session['user_chat_history'].append(user_entry)

    # Convert full history to Gemini-compatible format
    raw_history = session['user_chat_history']

    def convert_to_gemini_format(entry):
        if 'parts' in entry:
            return entry
        return {
            "role": entry.get("role", "user"),
            "parts": [{"text": entry.get("content", "")}]
        }

    history = [convert_to_gemini_format(entry) for entry in raw_history]

    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_message)
        bot_response = response.text

        # Add response to history
        session['user_chat_history'].append({
            "role": "model", "parts": [{"text": bot_response}]
        })
        session.modified = True

        # Optionally save to DB
        if logged_in and user_id:
            save_user_chat_history(user_id, session['user_chat_history'])

        return jsonify({"response": bot_response})

    except Exception as e:
        logging.error(f"Gemini chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

from flask_dance.contrib.google import make_google_blueprint, google

# Add Google OAuth config (use environment variables for security)
GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")

if not GOOGLE_OAUTH_CLIENT_ID or not GOOGLE_OAUTH_CLIENT_SECRET:
    raise ValueError("GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET must be set in the environment")

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Only for local development

google_bp = make_google_blueprint(
    client_id=GOOGLE_OAUTH_CLIENT_ID,
    client_secret=GOOGLE_OAUTH_CLIENT_SECRET,
    scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile"
    ],
    redirect_url="/login/google/authorized"
)
app.register_blueprint(google_bp, url_prefix="/login")

@app.route("/login/google")
def login_google():
    logging.info("/login/google route hit. google.authorized=%s", google.authorized)
    if not google.authorized:
        logging.info("User not authorized with Google. Redirecting to google.login.")
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    logging.info("Google userinfo response: %s", resp.text if resp else None)
    if not resp.ok:
        logging.error("Failed to fetch user info from Google. Status: %s", resp.status_code)
        return "Failed to fetch user info from Google.", 400
    user_info = resp.json()
    email = user_info["email"]
    username = user_info.get("name", email.split("@")[0])
    user_id = user_info.get("id", email)
    session["logged_in"] = True
    session["username"] = username
    session["user_id"] = user_id
    session["email"] = email
    session.permanent = True
    logging.info(f"Google login: {username} ({email}) user_id={user_id}")
    # Optionally: create user in DynamoDB if not exists
    if users_table:
        try:
            response = users_table.get_item(Key={"username": username})
            if "Item" not in response:
                users_table.put_item(Item={
                    "username": username,
                    "user_id": user_id,
                    "email": email,
                    "password_hash": "google-oauth"
                })
                logging.info(f"Created new Google user in DB: {username}")
        except Exception as e:
            logging.error(f"Error creating Google user in DB: {e}")
    return redirect(url_for("profile"))

@app.route("/login/google/authorized")
def google_authorized():
    logging.info("/login/google/authorized route hit. google.authorized=%s", google.authorized)
    if not google.authorized:
        logging.info("User not authorized with Google. Redirecting to google.login.")
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    logging.info("Google userinfo response: %s", resp.text if resp else None)
    if not resp.ok:
        logging.error("Failed to fetch user info from Google. Status: %s", resp.status_code)
        return "Failed to fetch user info from Google.", 400
    user_info = resp.json()
    email = user_info["email"]
    username = user_info.get("name", email.split("@")[0])
    user_id = user_info.get("id", email)
    session["logged_in"] = True
    session["username"] = username
    session["user_id"] = user_id
    session["email"] = email
    session.permanent = True
    logging.info(f"Google authorized: {username} ({email}) user_id={user_id}")
    # Optionally: create user in DynamoDB if not exists
    if users_table:
        try:
            response = users_table.get_item(Key={"username": username})
            if "Item" not in response:
                users_table.put_item(Item={
                    "username": username,
                    "user_id": user_id,
                    "email": email,
                    "password_hash": "google-oauth"
                })
                logging.info(f"Created new Google user in DB: {username}")
        except Exception as e:
            logging.error(f"Error creating Google user in DB: {e}")
    return redirect(url_for("profile"))

if __name__ == '__main__':
    print("🚀 Running locally at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

from dotenv import load_dotenv
import os
import secrets
from datetime import timedelta
import time
import json
import logging
from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from typing import List, Dict, Any, Optional

# --- Load environment variables at the very top ---
load_dotenv()

# --- Load hymn data ---
hymn_data = []
try:
    with open("hymn_book.json", "r", encoding="utf-8") as f:
        hymn_data = json.load(f)
    logging.info(f"Successfully loaded {len(hymn_data)} hymns from hymn_book.json")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logging.error(f"Could not load or parse hymn_book.json: {e}.")
    # The app can still run, but hymn search will not work.

def find_relevant_hymns(query: str, top_n: int = 3) -> List[Dict]:
    """
    Find relevant hymns using simple text matching.
    Returns a list of hymns that best match the query.
    """
    if not hymn_data:
        return []
    
    # Convert query to lowercase for case-insensitive matching
    query = query.lower()
    
    # Score each hymn based on matches in title and lyrics
    scored_hymns = []
    for hymn in hymn_data:
        score = 0
        title = hymn.get('title', '').lower()
        lyrics = ' '.join(hymn.get('lyrics', [])).lower()
        
        # Check title matches
        if query in title:
            score += 3  # Higher weight for title matches
        
        # Check lyrics matches
        if query in lyrics:
            score += 1
            
        # Add hymn to results if it has any matches
        if score > 0:
            scored_hymns.append((score, hymn))
    
    # Sort by score and return top N results
    scored_hymns.sort(reverse=True, key=lambda x: x[0])
    return [hymn for score, hymn in scored_hymns[:top_n]]

def format_hymn_for_context(hymn: dict) -> str:
    """Formats a single hymn's data into a readable string for the LLM context."""
    lyrics_text = "\n".join(hymn.get('lyrics', []))
    return (f"Hymn Number: {hymn.get('number', 'N/A')}\n"
            f"Title: {hymn.get('title', 'N/A')}\n"
            f"Author: {hymn.get('author', 'N/A')}\n"
            f"Composer: {hymn.get('composer', 'N/A')}\n"
            f"Lyrics:\n{lyrics_text}")


def load_persona_from_file(filepath):
    """Loads persona instructions and examples from a local JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            persona_data = json.load(f)
            logging.info(f"Successfully loaded persona from file: {filepath}")
            return persona_data
    except FileNotFoundError:
        logging.warning(f"Persona file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in persona file: {filepath}")
        return None


def load_chat_examples_from_txt(filepath: str) -> List[Dict[str, str]]:
    """
    Loads chat examples from a text file, where each pair of lines
    represents a user and bot message.
    """
    examples = []
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
            for i in range(0, len(lines) - 1, 2):
                user_message = lines[i].strip()
                bot_message = lines[i + 1].strip()
                examples.append({"user": user_message, "bot": bot_message})
        logging.info(f"Successfully loaded chat examples from: {filepath}")
        return examples
    except FileNotFoundError:
        logging.warning(f"Chat examples file not found: {filepath}")
        return []
    except Exception as e:
        logging.error(f"Error loading chat examples from {filepath}: {e}")
        return []

# Get the API key before importing genai
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment")

# Configure Gemini API
import google.generativeai as genai
genai.configure(api_key=api_key)

# Try different model options
model_options = ['gemini-pro-latest', 'gemini-pro']

# Get available models
try:
    available_models = [m.name for m in genai.list_models()]
    logging.info(f"Available models: {available_models}")
except Exception as e:
    logging.warning(f"Could not list models: {e}")
    available_models = []
chat_model = None

for model_name in model_options:
    try:
        chat_model = genai.GenerativeModel(model_name)
        logging.info(f"Successfully initialized model: {model_name}")
        break
    except Exception as e:
        logging.warning(f"Failed to initialize {model_name}: {e}")
        continue

if not chat_model:
    logging.error("Failed to initialize any chat model")
    raise ValueError("No suitable chat model available")

# --- Continue with the rest of your app setup ---

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

logging.info(f"Current working directory: {os.getcwd()}")
# --- Session Cookie Settings for OAuth reliability ---
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True only if using HTTPS

# --- Configuration from Environment Variables ---
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))
if app.secret_key == 'your_flask_secret_key_here':
    logging.warning("FLASK_SECRET_KEY is not set or is default. Please set a strong secret key in your environment.")
app.permanent_session_lifetime = timedelta(days=30)

# --- Application Settings ---
MAX_GUEST_MESSAGES = 40

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log environment variables (excluding sensitive values)
logging.info(f"FLASK_SECRET_KEY loaded: {'*' * len(app.secret_key) if app.secret_key else 'None'}")
logging.info(f"GOOGLE_API_KEY loaded: {'Set' if api_key else 'Not Set'}")
logging.info(f"MAX_GUEST_MESSAGES set to: {MAX_GUEST_MESSAGES}")

# --- Local Storage Functions ---

def _ensure_data_dir():
    """Ensures the data directory exists"""
    os.makedirs('data', exist_ok=True)

def _read_json_file(path: str) -> Dict:
    """Reads a JSON file and returns its content, or an empty dict if it doesn't exist."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _write_json_file(path: str, data: Dict):
    """Writes a dictionary to a JSON file."""
    _ensure_data_dir()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except IOError as e:
        logging.error(f"Could not write to file {path}: {e}")

def load_user_chat_history(user_id: str) -> List[Dict[str, Any]]:
    """Loads chat history for a given user from local storage."""
    try:
        history_file = os.path.join('data', f'chat_history_{user_id}.json')
        history = _read_json_file(history_file).get('history', [])
        if history:
            logging.info(f"Loaded chat history for user: {user_id}. Length: {len(history)}")
        return history
    except Exception as e:
        logging.error(f"Error loading chat history for user {user_id}: {e}")
        return []

def save_user_chat_history(user_id: str, history: List[Dict[str, Any]]) -> bool:
    """Saves chat history for a given user to local storage."""
    try:
        history_file = os.path.join('data', f'chat_history_{user_id}.json')
        _write_json_file(history_file, {'history': history})
        logging.info(f"Saved chat history for user: {user_id}. Length: {len(history)}")
        return True
    except Exception as e:
        logging.error(f"Error saving chat history for user {user_id}: {e}")
        return False

def get_user(username: str) -> Optional[Dict]:
    """Retrieves a user by username from local storage."""
    users_file = os.path.join('data', 'users.json')
    users = _read_json_file(users_file)
    return users.get(username)

def create_user(username: str, email: str, password_hash: str, user_id: str) -> bool:
    """Creates a new user in local storage."""
    users_file = os.path.join('data', 'users.json')
    users = _read_json_file(users_file)
    if username in users:
        return False  # User already exists
    
    users[username] = {
        'username': username,
        'password_hash': password_hash,
        'user_id': user_id,
        'email': email,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    _write_json_file(users_file, users)
    return True

def create_google_user_if_not_exists(username: str, email: str, user_id: str):
    """Creates a Google OAuth user if they don't exist."""
    users_file = os.path.join('data', 'users.json')
    users = _read_json_file(users_file)
    # Check if user with this email or username exists
    if not any(u.get('email') == email or k == username for k, u in users.items()):
        logging.info(f"Creating new Google user: {username}")
        users[username] = {
            "username": username,
            "user_id": user_id,
            "email": email,
            "password_hash": "google-oauth"
        }
        _write_json_db(USERS_DB_PATH, users_db)
    else:
        logging.info(f"No chat history found for user: {user_id}")

# Load persona on app startup
persona_file_path = os.path.join(os.path.dirname(__file__), "persona.json")
persona = load_persona_from_file(persona_file_path)

if persona:
    logging.info("Persona loaded successfully.")
else:
    logging.warning("Failed to load persona. Chatbot might not function as expected without a persona.")

# Initialize the Generative Model (using 1.5 Flash as it's efficient for chat)
model = chat_model

# --- Persona instruction moved into the beginning of the history ---
# This serves as a base for the chatbot's identity and behavior
initial_persona_prompt = []
if persona:
    # Use the detailed instructions from the persona file
    if persona.get("instructions"):
        initial_persona_prompt.extend([
            {
                "role": "user", "parts": [{"text": persona["instructions"] }],
            },
            {
                "role": "model", "parts": [{"text": f"Okay, I understand. I am {persona.get('name', 'Musik-Assist')}, and I will respond according to my persona."}]},
        ])
    # Add the hymn data constraint
    if persona.get("hymn_data_constraint"):
        initial_persona_prompt.extend([
            {"role": "user", "parts": [{"text": persona['hymn_data_constraint']}]},
            {"role": "model", "parts": [{"text": "I understand. I will use only the context provided with each question to answer questions about hymns."}]}
        ])
else:
    # Fallback to the old hardcoded prompt if persona.json is missing
    logging.warning("Persona not found or incomplete, using hardcoded fallback prompts.")
    initial_persona_prompt = [
        {
            "role": "user", "parts": [{"text": "You are Musik Assistant, a helpful chat assistant created by Ebenezer. You will always respond as Musik Assistant and adhere to the persona provided in your training data."}],
        },
        {
            "role": "model", "parts": [{"text": "Okay, I understand. I am Musik Assistant, and I will respond as a helpful chat assistant created by Ebenezer, adhering to my defined persona."}]},
        {"role": "user", "parts": [{"text": "For questions about hymns, you will be provided with relevant hymn information as context. You must use ONLY this provided context to answer. Do not use any other source for hymn information. If no relevant context is found for a user's query, or if the context doesn't answer the question, politely state that you can't find information on that topic and ask if they need help with something else."}]},
        {"role": "model", "parts": [{"text": "I understand. I will use only the context provided with each question to answer questions about hymns."}]}
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
        if 'logged_in' not in session:
            # For API endpoints that expect JSON
            is_api_request = request.path.startswith('/api/') or \
                             request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
                             (request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html)

            if is_api_request:
                logging.warning(f"Unauthorized API access attempt to {request.path}.")
                return jsonify({"error": "Unauthorized. Please log in."}), 401
            else:
                # For regular page views, redirect to home where login can occur
                logging.warning(f"Unauthorized page access to {request.path}. Redirecting to home.")
                return redirect(url_for('index'))

        return f(*args, **kwargs)
    return decorated_function

@app.context_processor
def inject_nav_links():
    """Injects navigation links into the context of all templates."""
    nav_links = [
        {'url': url_for('index'), 'text': 'Home'},
        {'url': url_for('about'), 'text': 'About'},
    ]
    if session.get('logged_in'):
        nav_links.extend([
            {'url': url_for('profile'), 'text': 'Profile'},
            {'url': url_for('settings'), 'text': 'Settings'},
        ])
    return dict(nav_links=nav_links)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    email = request.json.get('email') # Get email from request
    password = request.json.get('password')

    if not username or not email or not password: # Ensure email is also required for registration
        logging.warning("Registration attempt with missing username, email, or password.")
        return jsonify({"error": "Username, email, and password are required"}), 400

    try:
        # Check if username already exists
        if get_user(username):
            logging.warning(f"Registration attempt for existing username: {username}.")
            return jsonify({"error": "Username already exists"}), 409 # 409 Conflict

        # Hash password
        password_hash = generate_password_hash(password)
        user_id = secrets.token_hex(16)

        if create_user(username, email, password_hash, user_id):
            logging.info(f"User '{username}' registered successfully with user_id: {user_id}")
            return jsonify({"message": "Registration successful"}), 201 # 201 Created
        else:
            # This case should be caught by the get_user check, but as a fallback
            return jsonify({"error": "Username already exists"}), 409

    except Exception as e:
        logging.error(f"Error during registration for user '{username}': {e}", exc_info=True)
        return jsonify({"error": "Registration failed due to server error."}), 500

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    try:
        user_data = get_user(username)

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

def _filter_history_for_display(chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes initial system/persona prompts from chat history for frontend display."""
    if not chat_history:
        return []

    # Create a set of persona messages for efficient lookup
    persona_messages = set()
    for prompt in initial_persona_prompt:
        text = prompt['parts'][0]['text']
        persona_messages.add((prompt['role'], text))

    display_history = []
    for msg in chat_history:
        message_text = ""
        if msg.get('parts') and isinstance(msg['parts'], list) and len(msg['parts']) > 0:
            first_part = msg['parts'][0]
            if isinstance(first_part, dict) and 'text' in first_part:
                message_text = first_part['text']
            elif isinstance(first_part, str):
                message_text = first_part
        
        if (msg['role'], message_text) not in persona_messages:
            # Append to display_chat_history ensuring the correct format
            display_history.append({'role': msg['role'], 'parts': [message_text]})
    
    return display_history

@app.route('/check_login_status', methods=['GET'])
def check_login_status():
    if 'user_id' in session and session.get('logged_in'):
        user_id = session['user_id']
        username = session['username'] # Get username directly from session

        # Ensure chat history is loaded into session if not already present (e.g., on first page load or session restore)
        if 'user_chat_history' not in session:
            session['user_chat_history'] = get_gemini_chat_session_history()
            logging.info(f"Chat history re-initialized/loaded into session for user {user_id}.")

        display_chat_history = _filter_history_for_display(session['user_chat_history'])
        
        logging.info(f"Login status check: User '{username}' is logged in. History length for display: {len(display_chat_history)}")
        return jsonify({
            "logged_in": True, 
            "username": username,
            "user_id": user_id, # Frontend might need user_id for client-side distinctions
            "email": session.get('email'), # Get email from session, if available
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

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_message = request.json.get('message')
    if not user_message:
        logging.warning("Chat attempt with no message provided.")
        return jsonify({"error": "No message provided"}), 400

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
            session["user_chat_history"] = get_gemini_chat_session_history()
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

    try:
        # Find relevant hymns using simple text matching
        relevant_hymns = find_relevant_hymns(user_message, top_n=3)

        # 3. Construct the prompt with context
        final_prompt = user_message
        if relevant_hymns:
            context_string = "\n\n---\n\n".join([format_hymn_for_context(hymn) for hymn in relevant_hymns])
            final_prompt = (
                "Based on the following hymn information, please answer my question.\n\n"
                "---CONTEXT---\n"
                f"{context_string}\n"
                "---END CONTEXT---\n\n"
                f"QUESTION: {user_message}"
            )
            logging.info(f"Found {len(relevant_hymns)} relevant hymns for the query.")
        else:
            logging.info("No relevant hymns found via vector search for the query.")

        # Add user message to history (using the augmented prompt with context)
        session['user_chat_history'].append({"role": "user", "parts": [{"text": final_prompt}]})

        # 4. Get response from the model
        # Pass all but the last message (the current user prompt) as history
        chat_history_for_model = session['user_chat_history'][:-1]
        chat_session = model.start_chat(history=chat_history_for_model)
        response = chat_session.send_message(final_prompt)  # Send only the new message with context
        bot_response = response.text

        # 5. Add bot response to history and save
        session['user_chat_history'].append({"role": "model", "parts": [{"text": bot_response}]})
        session.modified = True
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
    create_google_user_if_not_exists(username, email, user_id)
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
    create_google_user_if_not_exists(username, email, user_id)
    return redirect(url_for("profile"))

if __name__ == '__main__':
    print("🚀 Running locally at http://127.0.0.1:5000")
    # Use a dynamic port for the application
    app.run(debug=True, host='127.0.0.1', port=5000 if os.getenv("PORT") is None else int(os.getenv("PORT")))

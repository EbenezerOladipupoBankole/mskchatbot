from dotenv import load_dotenv
import os
import secrets
from datetime import timedelta
import time
import json
import logging
from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from flask_session import Session
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from typing import List, Dict, Any, Optional
import tempfile
import numpy as np

# --- Load environment variables at the very top ---
load_dotenv()

# --- Flask Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))

# Configure Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(tempfile.gettempdir(), 'flask_session')
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=31)
Session(app)

# --- Simulated Google Search Tool ---
class GoogleSearch:
    """A simulated class for performing Google searches."""
    def search(self, queries: List[str]):
        """Simulates a web search and returns placeholder results."""
        logging.info(f"Simulating web search for queries: {queries}")
        # In a real application, you would use a library like `requests`
        # to call a search API and return real results.
        return {q: [{"title": "Simulated Search Result", "snippet": "This is a placeholder for web search results. Integrating a real search API would provide dynamic, real-time information."}] for q in queries}

google_search = GoogleSearch()

# --- Load Hymn Data and Embeddings ---
hymn_data_raw = []
hymns_by_number = {}
hymn_embeddings = {}

# Define data directory relative to this script
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

try:
    with open(os.path.join(DATA_DIR, "hymn_book.json"), "r", encoding="utf-8") as f:
        hymn_data_raw = json.load(f)
        hymns_by_number = {hymn['number']: hymn for hymn in hymn_data_raw if 'number' in hymn}
    logging.info(f"Successfully loaded {len(hymn_data_raw)} hymns.")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logging.error(f"Could not load or parse hymn_book.json from {DATA_DIR}: {e}.")

try:
    with open(os.path.join(DATA_DIR, "hymn_embeddings.json"), "r", encoding="utf-8") as f:
        hymn_embeddings = json.load(f)
    logging.info(f"Successfully loaded {len(hymn_embeddings)} hymn embeddings.")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logging.error(f"Could not load or parse hymn_embeddings.json: {e}. Semantic search will not work.")
    logging.error("Please run generate_embeddings.py to create the embeddings file.")

def find_relevant_hymns(query: str, top_n: int = 3) -> List[Dict]:
    """
    Finds relevant hymns using semantic search with vector embeddings.
    Returns a list of the original hymn dictionaries.
    """
    if not hymn_embeddings or not hymns_by_number:
        logging.warning("Hymn data or embeddings not loaded. Cannot perform search.")
        return []

    # Generate embedding for the user's query
    query_embedding = genai.embed_content(
        model='models/embedding-001',
        content=query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']

    scored_hymns = []
    for hymn_num, hymn_vec in hymn_embeddings.items():
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, hymn_vec) / (np.linalg.norm(query_embedding) * np.linalg.norm(hymn_vec))
        
        if hymn_num in hymns_by_number:
            scored_hymns.append((similarity, hymns_by_number[hymn_num]))

    # Sort by score in descending order and return the top N original hymn dicts
    scored_hymns.sort(reverse=True, key=lambda x: x[0])
    
    # Filter out low-scoring results to avoid irrelevant context
    # This threshold may need tuning.
    relevant_hymns = [hymn for score, hymn in scored_hymns if score > 0.65]
    
    return relevant_hymns[:top_n]

def format_hymn_for_context(hymn: dict) -> str:
    """Formats a single hymn's data into a readable string for the LLM context."""
    lyrics_text = "\n".join(hymn.get('lyrics', []))
    musical_info = hymn.get('musical_info', {})
    
    # Format musical information
    musical_details = (
        f"Musical Information:\n"
        f"Key Signature: {musical_info.get('key_signature', 'N/A')}\n"
        f"Time Signature: {musical_info.get('time_signature', 'N/A')}\n"
        f"Tempo: {musical_info.get('tempo', 'N/A')}\n"
        f"Meter: {musical_info.get('meter', 'N/A')}\n"
    )
    
    # Format vocal ranges if available
    ranges = musical_info.get('musical_setting', {}).get('ranges', {})
    if ranges:
        musical_details += (
            f"\nVocal Ranges:\n"
            f"Soprano: {ranges.get('soprano', 'N/A')}\n"
            f"Alto: {ranges.get('alto', 'N/A')}\n"
            f"Tenor: {ranges.get('tenor', 'N/A')}\n"
            f"Bass: {ranges.get('bass', 'N/A')}\n"
        )
    
    return (f"Hymn #{hymn.get('number', 'N/A')}: {hymn.get('title', 'N/A')}\n"
            f"Author: {hymn.get('author', 'N/A')}\n"
            f"Composer: {hymn.get('composer', 'N/A')}\n\n"
            f"{musical_details}\n"
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

# Get the API key before importing genai
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment")

# Configure Gemini API
genai.configure(api_key=api_key)

# Try different model options
model_options = ['gemini-1.5-flash', 'gemini-pro']

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
            # Store minimal session data
            session.clear()  # Clear any existing session data
            session['user_id'] = user_data['user_id']
            session['logged_in'] = True
            session['login_time'] = int(time.time())
            
            # Log the login
            logging.info(f"User '{username}' logged in successfully.")
            
            # Return minimal response
            return jsonify({
                "message": "Login successful",
                "username": user_data['username']
            }), 200
        else:
            # Log failed attempt
            logging.warning(f"Login failed for user '{username}'. Invalid credentials.")
            session.clear()  # Clear all session data
            return jsonify({"error": "Invalid credentials"}), 401
            
    except Exception as e:
        logging.error(f"Error during login for user '{username}': {e}", exc_info=True)
        session.clear()  # Clear session data on error
        return jsonify({"error": "Login failed due to server error."}), 500

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    username = session.get('username', 'Unknown')
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('guest_message_count', None) # Clear guest count on logout
    session.permanent = False
    logging.info(f"User '{username}' logged out.")
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

        # Load history directly from storage for display
        user_chat_history = load_user_chat_history(user_id)
        display_chat_history = _filter_history_for_display(user_chat_history)
        
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

    # Load or initialize chat history directly, not from session
    if logged_in and user_id:
        chat_history = load_user_chat_history(user_id)
        if not chat_history: # If user has no history, initialize it
            chat_history = get_gemini_chat_session_history() # This initializes with persona
    else:
        # For guests, get history from the session object
        if 'chat_history' not in session:
            session['chat_history'] = get_gemini_chat_session_history() # Initialize with persona
        chat_history = session['chat_history']

    # Guest message limit logic
    if not logged_in:
        session.permanent = True # Ensure guest session persists
        guest_message_count = session.get('guest_message_count', 0)
        if guest_message_count >= MAX_GUEST_MESSAGES:
            return jsonify({
                "error": f"You've reached the limit of {MAX_GUEST_MESSAGES} free messages. Please log in.",
                "code": "LIMIT_EXCEEDED"
            }), 403
        session['guest_message_count'] = guest_message_count + 1
        session.modified = True

    try:
        # 1. Perform Web Search for broad context
        search_queries = [user_message]
        web_search_results = google_search.search(queries=search_queries)
        
        # 2. Find relevant hymns using local vector search
        relevant_hymns = find_relevant_hymns(user_message, top_n=3)

        # 3. Construct the prompt with context
        final_prompt = user_message
        context_parts = []

        # Add web search context
        if web_search_results:
            web_context = "\n".join([f"Title: {item['title']}\nSnippet: {item['snippet']}" for query_results in web_search_results.values() for item in query_results])
            context_parts.append(f"---WEB SEARCH RESULTS---\n{web_context}")
            logging.info("Added web search results to context.")

        # Add local hymn context
        if relevant_hymns:
            hymn_context = "\n\n---\n\n".join([format_hymn_for_context(hymn) for hymn in relevant_hymns])
            context_parts.append(f"---HYMN BOOK DATA---\n{hymn_context}")
            logging.info(f"Found {len(relevant_hymns)} relevant hymns for the query.")

        if context_parts:
            full_context = "\n\n".join(context_parts)
            final_prompt = (
                "Based on the following information, please answer my question. Synthesize information from both web search results and the hymn book data if available. Do not use any Markdown formatting in your response. Present verses with simple line breaks.\n\n"
                f"{full_context}\n\n"
                f"---END OF CONTEXT---\n\n"
                f"QUESTION: {user_message}"
            )

        # Add user message to history (using the augmented prompt with context)
        chat_history.append({"role": "user", "parts": [{"text": final_prompt}]})

        # 4. Get response from the model
        chat_history_for_model = chat_history[:-1]
        chat_session = model.start_chat(history=chat_history_for_model)
        response = chat_session.send_message(final_prompt)  # Send only the new message with context
        bot_response = response.text

        # Log if the bot couldn't find specific information that was asked for
        if (("don't know" in bot_response.lower() or "don't have information" in bot_response.lower()) and
                ("composer" in user_message.lower() or "author" in user_message.lower())):
            hymn_title_or_num = relevant_hymns[0]['title'] if relevant_hymns else "unknown"
            logging.warning(f"Missing data: User asked for composer/author for hymn '{hymn_title_or_num}', which is not in the dataset.")

        # 5. Add bot response to history and save
        chat_history.append({"role": "model", "parts": [{"text": bot_response}]})
        if logged_in and user_id:
            save_user_chat_history(user_id, chat_history)
        else:
            request.session['chat_history'] = chat_history

        return {"response": bot_response}

    except Exception as e:
        logging.error(f"Gemini chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    print("🚀 Running locally at http://127.0.0.1:8000")
    uvicorn.run(app, host='127.0.0.1', port=8000)

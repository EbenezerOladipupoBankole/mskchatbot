# Musik Chatbot

A Flask-based chatbot web application for hymn lookup, chat, and user authentication, featuring Google OAuth login and DynamoDB integration.

## Features
- Chatbot powered by Google Gemini API
- Hymn lookup from local JSON file
- User registration and login (email/password)
- Google OAuth login (secure, using environment variables)
- DynamoDB for user and chat history storage
- Persona and example Q&A loaded from S3
- Guest message limit
- Modern frontend (HTML/CSS/JS)

## Setup

### Prerequisites
- Python 3.10+
- AWS account (for DynamoDB and S3)
- Google Cloud project (for OAuth and Gemini API)

### Installation
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd mskchatbot
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set up your `.env` file with the following keys:
   ```ini
   FLASK_SECRET_KEY=your-very-secret-key
   GOOGLE_API_KEY=your-gemini-api-key
   GOOGLE_OAUTH_CLIENT_ID=your-google-oauth-client-id
   GOOGLE_OAUTH_CLIENT_SECRET=your-google-oauth-client-secret
   AWS_ACCESS_KEY_ID=your-aws-access-key-id
   AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
   AWS_DEFAULT_REGION=your-aws-region
   S3_BUCKET_NAME=your-s3-bucket-name
   ```

### Google OAuth Setup
- Register your app in Google Cloud Console
- Add redirect URI: `http://127.0.0.1:5000/login/google/authorized`
- Use client ID/secret in `.env`

### AWS Setup
- Create DynamoDB tables: `MusicChatbotUsers`, `MusicChatHistory`
- Upload `persona.json` to your S3 bucket

## Running the App
```sh
python musichat.py
```
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Folder Structure
- `musichat.py` — Main Flask backend
- `hymn_book.json` — Hymn data
- `persona.json` — Persona and example Q&A
- `templates/` — HTML templates
- `static/` — CSS and JS files
- `images/` — App images

## Security Notes
- **Never hardcode secrets in code.** Use environment variables.
- For production, use HTTPS and a production WSGI server.

## License
MIT

"""

This file should be very thin. It will include things that will be imported throughout the application

 It should include the following:
- App
- Database
- A cache
- Any logging handlers
- Task Queues
- etc

"""
from flask import Flask
from sqlalchemy import create_engine

import os


# TODO: Change this to loading via environment variables
SQLALCHEMY_DATABASE_URI = 'postgresql://flaskapp:foo@userdb:5432/flaskapp'


# Application Setup
app = Flask(__name__)
app.secret_key = os.urandom(12) # Generic key for dev purposes only
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', './uploads')


# Database setup
db = create_engine(SQLALCHEMY_DATABASE_URI)

# Any other setup add below

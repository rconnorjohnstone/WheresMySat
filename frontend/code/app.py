"""
Authors: Hunter Mellema, Connor Johnstone, Chelsea Thangavelu
Description: Defines the base app for use by a WSGI compliant server

Notes:
This file should be very thin. It will include things that will be imported throughout the application

 It should include the following:
- App
- Run the database setup
- A cache
- Any logging handlers
- Task Queues
- etc

"""
######### BEGIN IMPORTS ###############
# standard library imports
import os

# third party imports
from flask import Flask
from sqlalchemy import create_engine

# local imports
from errors import bp as errors_bp

######### END IMPORTS ################


# TODO: Change this to loading via environment variables
SQLALCHEMY_DATABASE_URI = 'postgresql://flaskapp:foo@userdb:5432/flaskapp'


# Application Setup
app = Flask(__name__)
app.secret_key = os.urandom(12) # Generic key for dev purposes only
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', './uploads')


# Database setup
db = create_engine(SQLALCHEMY_DATABASE_URI)


# register blueprints
app.register_blueprint(errors_bp)


# Any other setup add below

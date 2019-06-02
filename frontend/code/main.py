"""
Author: Hunter Mellema, Connor Johnstone, Chelsea Thangavelu
Description: Main script for the application. Point a WGSI compliant server to this file to
    run the application

"""
from app import app, db
from scripts.tabledef import *
from views import *


def create_tables():
    """ Creates a database table for each model in models.py if it does not exist """
    Base.metadata.create_all(db)


if __name__ == '__main__':
    create_tables()
    serve(app, host= '0.0.0.0', port=5000)

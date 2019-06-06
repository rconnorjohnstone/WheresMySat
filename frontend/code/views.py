# -*- coding: utf-8 -*-
"""
Author: Connor Johnstone, Hunter Mellema, and Chelsea Thangavelu
Description: Defines the routes and page rendering for the application
Notes:views imports app, auth, and models, but none of these import views

"""
####### Begin Imports ############

# standard library imports
import json
import sys
import os
from functools import wraps

# third party imports
from flask import Flask, redirect, url_for, flash, render_template, request, session
from waitress import serve
from werkzeug import secure_filename
from jinja2 import Template
from random import randint

# local app imports
from app import app
from scripts import tabledef
from scripts import forms
from scripts import helpers

####### END IMPORTS ##############


# ======== Constants ========================================================= #
ALLOWED_EXTENSIONS = set(['csv', 'txt'])


# ======== Auth Validation  ================================================== #
def auth_required(fxn):
    """ Verifies that user is logged in. If not, routes them to the login page

    """
    @wraps(fxn)
    def auth_req_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.url))
        else:
            return fxn(*args, **kwargs)
    return auth_req_function


def skip_to_home(fxn):
    """ Skips past page straight to home page if logged in

    """
    @wraps(fxn)
    def skipped_page_fxn(*arg, **kwargs):
        if session.get('logged_in'):
            return redirect(url_for('home'))
        else:
            return fxn(*arg, **kwargs)
    return skipped_page_fxn


# ======== Routing =========================================================== #

# -------- Home Page --------------------------------------------------------- #
@app.route('/home', methods=['GET', 'POST'])
@auth_required
def home():
    user = helpers.get_user()
    return render_template("home.html", user=user)


# -------- Landing Page ------------ ----------------------------------------- #
@app.route('/', methods=['GET', 'POST'])
@skip_to_home
def landing_page():
    return render_template('out_facing/landing_page.html')


# -------- Login ------------------------------------------------------------- #
@app.route('/login', methods=["GET", "POST"])
def login():
    form = forms.LoginForm(request.form)

    if request.method == 'POST':
        username = request.form['username'].lower()
        password = request.form['password']
        if form.validate():
            if helpers.credentials_valid(username, password):
                session['logged_in'] = True
                session['username'] = username
                user = helpers.get_user()
                return render_template('home.html', user=user), 200

            else:
                flash('Wrong Username or Password', 'error')
                return render_template("out_facing/login.html"), 401

        else:
            flash('Both fields required', 'error')
            return render_template("out_facing/login.html"), 400

    return render_template("out_facing/login.html")


# -------- Signup ---------------------------------------------------------- #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if session.get('logged_in'):
        return redirect(url_for('home'))
    else:
        form = forms.SignupForm(request.form)

        if request.method == 'POST':
            username = request.form['username'].lower()
            password = helpers.hash_password(request.form['password'])
            email = request.form['email']

            if form.validate():
                if not helpers.username_taken(username):
                    helpers.add_user(username, password, email)
                    session['logged_in'] = True
                    session['username'] = username
                    return redirect(url_for('home'))
                flash('Username taken', 'error')
                return render_template("out_facing/signup.html"), 401
            flash("Missing a required field", 'error')
            return render_template("out_facing/signup.html"), 400
        return render_template("out_facing/signup.html")


#-------- Log Out ------------------------------------------------------------ #
@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('landing_page'))


# -------- Settings ---------------------------------------------------------- #
@app.route('/settings', methods=['GET', 'POST'])
@auth_required
def settings():
    if request.method == 'POST':
        password = request.form['password']
        if password != "":
            password = helpers.hash_password(password)
        email = request.form['email']
        helpers.change_user(password=password, email=email)
        return json.dumps({'status': 'Saved'})
    user = helpers.get_user()
    return render_template('settings.html', user=user)


#--------- File Upload ------------------------------------------------------- #
def upload_type_check(filename):
    """ Checks that upload file type has the required file extension

    Args:
        filename (string): name of file to check

    """
    condition = ('.' in filename) and (filename.rsplit('.', 1)[1].lower()
                                       in ALLOWED_EXTENSIONS)
    return condition

@app.route('/uploader', methods = ['POST'])
def uploader():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file!', 'warning')
        if file:
            if not upload_type_check(file.filename):
                flash('Wrong file type. Only accepts: {}'.format(ALLOWED_EXTENSIONS),
                      'warning'
                )
            else:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                flash("File upload success!", 'success')
        return redirect(url_for('login'))


#---------- Filtering interface ---------------------------------------------- #

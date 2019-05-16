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


# ======== Routing =========================================================== #
# -------- Login ------------------------------------------------------------- #
@app.route('/', methods=['GET', 'POST'])
def login():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = request.form['password']
            if form.validate():
                if helpers.credentials_valid(username, password):
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Login successful'})
                return json.dumps({'status': 'Invalid user/pass'})
            return json.dumps({'status': 'Both fields required'})
        return render_template('login.html', form=form)
    user = helpers.get_user()
    return render_template('home.html', user=user)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))


# -------- Signup ---------------------------------------------------------- #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = helpers.hash_password(request.form['password'])
            email = request.form['email']
            if form.validate():
                if not helpers.username_taken(username):
                    helpers.add_user(username, password, email)
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Signup successful'})
                return json.dumps({'status': 'Username taken'})
            return json.dumps({'status': 'User/Pass required'})
        return render_template('login.html', form=form)
    return redirect(url_for('login'))


# -------- Settings ---------------------------------------------------------- #
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if session.get('logged_in'):
        if request.method == 'POST':
            password = request.form['password']
            if password != "":
                password = helpers.hash_password(password)
            email = request.form['email']
            helpers.change_user(password=password, email=email)
            return json.dumps({'status': 'Saved'})
        user = helpers.get_user()
        return render_template('settings.html', user=user)
    return redirect(url_for('login'))

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


#---------- 404 Error Page --------------------------------------------------- #
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

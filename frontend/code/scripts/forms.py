# -*- coding: utf-8 -*-
"""
Description: Defines the structures of forms to be read from a web page
"""

from wtforms import Form, StringField, validators


class LoginForm(Form):
    username = StringField('Username:',
                            validators=[validators.required(),
                            validators.Length(min=1, max=30)])
    password = StringField('Password:',
                           validators=[validators.required(),
                           validators.Length(min=1, max=30)])
    email = StringField('Email:',
                        validators=[validators.optional(),
                        validators.Length(min=0, max=50)])

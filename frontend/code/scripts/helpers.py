# -*- coding: utf-8 -*-
from app import app
from scripts import tabledef
from flask import session
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from random import randint
import bcrypt


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    s = get_session()
    s.expire_on_commit = False
    try:
        yield s
        s.commit()
    except:
        s.rollback()
        raise
    finally:
        s.close()


def get_session():
    return sessionmaker(bind=tabledef.engine)()


def get_user():
    username = session['username']
    with session_scope() as s:
        user = s.query(tabledef.User).filter(tabledef.User.username.in_([username])).first()
        return user


def add_user(username, password, email):
    with session_scope() as s:
        u = tabledef.User(username=username, password=password, email=email)
        s.add(u)
        s.commit()


def change_user(**kwargs):
    username = session['username']
    with session_scope() as s:
        user = s.query(tabledef.User).filter(tabledef.User.username.in_([username])).first()
        for arg, val in kwargs.items():
            if val != "":
                setattr(user, arg, val)
        s.commit()


def hash_password(password):
    return bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())


def credentials_valid(username, password):
    with session_scope() as s:
        user = s.query(tabledef.User).filter(tabledef.User.username.in_([username])).first()
        if user:
            return bcrypt.checkpw(password.encode('utf8'), user.password)
        else:
            return False

def username_taken(username):
    with session_scope() as s:
        return s.query(tabledef.User).filter(tabledef.User.username.in_([username])).first()


#### Generate Cool Star effects ####
def gen_star_pos_list(nstars):
    """ Generates a random list of positions to place stars """
    pos_list = "{0}px {1}px #FFF;".format(randint(0,2000), randint(0,2000))

    for i in range(nstars):

        pos_list = '{0}px {1}px #FFF,'.format(randint(0,2000), randint(0,2000)) + pos_list

    return pos_list

# adds the start function to the globals so you can use it in a template
app.jinja_env.globals.update(gen_star_pos_list=gen_star_pos_list)
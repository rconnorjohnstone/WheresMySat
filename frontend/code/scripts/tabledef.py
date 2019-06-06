# -*- coding: utf-8 -*-
"""
Author: Hunter Mellema
Description: Defines the database models to use in the application

"""
# third party imports
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    username = Column(String(30), unique=True)
    password = Column(String(200))
    email = Column(String(30))

    def __repr__(self):
        return '<User %r>' % self.username

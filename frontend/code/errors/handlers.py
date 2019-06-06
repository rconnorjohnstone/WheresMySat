"""
Authors: Hunter Mellema
Description: Defines error handlers for HTTP error codes

Currently Defined Error Handlers:
- 404: page not found
- 500: Internal Server Error

"""
from flask import render_template, request
from errors import bp


#---------- 404 Error Page --------------------------------------------------- #
@bp.app_errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404


#---------- 500 Error Page --------------------------------------------------- #
@bp.app_errorhandler(500)
def server_error(e):
    return render_template('errors/500.html'), 500

FROM python:3.7.2-alpine

# install packages necessary to install and build python requirements
RUN apk update &&\
	apk add gcc &&\
	apk add libc-dev &&\
	apk add libffi-dev &&\
	apk add postgresql-dev

# copy over files to desired directory
COPY ./code/ /usr/app
WORKDIR /usr/app

# add all the required python packages
RUN pip install -r requirements.txt

# Create user groups for running the application
# also set the home directory of the user to the app directory
#RUN addgroup -S flaskapp &&\
#    adduser -S -D -h /usr/app flaskuser flaskapp &&\
#    chown -R flaskuser:flaskapp /usr/app

# switch user to the flask app user profile
#USER flaskuser:flaskapp

# map the default flask port!
EXPOSE 5000

# run the app!
CMD ["python", "main.py"]

# Use an official Python alpine runtime as a parent image
FROM python:3.9

# Make /opt/app directory, add postgres dep and upgrade pip
RUN mkdir -p /opt/app && \
  pip install --upgrade pip

# Project setup
WORKDIR /opt/app
COPY . /opt/app/

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run app.py when the container launches
CMD [ "python", "./app.py" ]
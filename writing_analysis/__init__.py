from flask import Flask
import os

print(os.listdir())

app = Flask(__name__)
app.secret_key = "b':\xca\x84\xe4q\xbf;w\xe8\xb3\x01\x1f/\x9a]X*(\x0c:\xd5\xd4\xae\x01'"

from .processml import *
from writing_analysis import routes

from .routes import routes as route_blueprint
app.register_blueprint(route_blueprint)
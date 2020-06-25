from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from writing_analysis import *
import json


routes = Blueprint('routes', __name__)

@routes.route('/')
def home():
    return render_template('home.html', result=False)

@app.route('/', methods=['POST'])
def result():
    text = request.form['text']

    if(text == ""):
        flash("Please enter something into the box.")
        return render_template('home.html', result=False)

    ans = processpost(text)
    
    for i in range(len(ans)):
        ans[i]['wordo'] = ans[i]['word']
        ans[i]['word'] = ans[i]['word'].replace(" ", "&nbsp;")
    print(ans)

    return render_template('home.html', result=True, txt=enumerate(ans))


        
    
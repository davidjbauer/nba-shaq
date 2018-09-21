#!/opt/local/bin/python3.6
# from flask import Flask, request
# from flask_restplus import Resource, Api
# from flask_sqlalchemy import SQLAlchemy
import flask
import flask_restplus as restplus
import pandas as pd
import os

player_df = pd.read_json("data/1980-2017-per100-labeled-nodup.json")
predict_df = pd.read_json("data/mvp-predict-2018-beta.json")
predict_df.sort_values(by="Probability", inplace=True, ascending=False)

basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)
app = flask.Flask(__name__)

def highlight_mvp(s):
    return ['background-color: #00FF00' if s.MVP == 1 else '' for v in s]

@app.route('/table/view/<int:year>')
def stat_view(year):
    p = player_df[player_df.Year == year].style.apply(highlight_mvp, axis=1).set_table_attributes('class = "dataframe table-sm table-bordered"')
    mytable = p.render()
    return flask.render_template("table_view.html", table=mytable, year=year)

@app.route('/table/json/<int:year>')
def stat_json(year):
    p = player_df[player_df.Year == year]
    j = p.to_json()
    return j

@app.route('/')
def index_html():
    return app.send_static_file('index.html')

@app.route('/pred/<int:year>')
def pred_table(year):
    if (year < 2018):
        return "SHAQ only provides predictions for the 2017-2018 season and later."
    if (year == 2018):
        mytable = predict_df.style.set_table_attributes('class = "dataframe table-sm table-bordered"').render()
        return flask.render_template("table_view.html", table=mytable, year=year)
    else:
        return "Check back later."


if (__name__ == '__main__'):
    app.run(debug = True)



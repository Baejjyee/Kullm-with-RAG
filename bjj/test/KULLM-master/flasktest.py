from flask import Flask
from test import core_function

app = Flask(__name__)

@app.route('/')
def index():
    return core_function()

if __name__ == '__main__':
    app.run(debug=True)

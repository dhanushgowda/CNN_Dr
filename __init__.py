from flask import Flask, jsonify, request
from flask import render_template
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("home.html")

@app.route("/run",methods=['GET'])
def run_eval():
    x = int(request.args.get('itr'))
    return render_template("op.html",op_val= x)


if __name__ == "__main__":
    app.run(host='0.0.0.0')

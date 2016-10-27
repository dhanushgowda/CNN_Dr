from flask import Flask, jsonify, request
from flask import render_template
from eval import begin_here
app = Flask(__name__)

@app.route("/")
def run_eval():
    # x = int(request.args.get('itr'))
    x, y  = begin_here()
    pred = [j[0] for j in x]
    actual = [j[1][0] for j in x]
    pred_cat = [b.argmax() for b in pred]
    print pred_cat
    print actual
    z=[]
    for i in range(0,len(y)-1):
        z.append([pred_cat[i],actual[i],y[i]])
    print '\n'
    print z
    return render_template("op.html",op_val= y,pred=pred_cat,actual=actual,data=z)

@app.route("/run",methods=['GET'])
def run_eval():
    # x = int(request.args.get('itr'))
    x, y  = begin_here()
    pred = [j[0] for j in x]
    actual = [j[1][0] for j in x]
    pred_cat = [b.argmax() for b in pred]
    print pred_cat
    print actual
    z=[]
    for i in range(0,len(y)-1):
        z.append([pred_cat[i],actual[i],y[i]])
    print '\n'
    print z
    return render_template("op.html",op_val= y,pred=pred_cat,actual=actual,data=z)


if __name__ == "__main__":
    app.run(host='0.0.0.0')

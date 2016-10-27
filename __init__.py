from flask import Flask, make_response, render_template
from eval import get_predictions
from functools import update_wrapper

app = Flask(__name__)


def nocache(f):
    def new_func(*args, **kwargs):
        resp = make_response(f(*args, **kwargs))
        resp.cache_control.no_cache = True
        return resp

    return update_wrapper(new_func, f)


@app.route("/")
@nocache
def run_eval():
    x, y = get_predictions()
    pred = [i[0] for i in x]
    actual = [i[1][0] for i in x]
    pred_cat = [b.argmax() for b in pred]
    # print pred_cat
    # print actual
    z = []
    for i in range(0, len(y) - 1):
        z.append([pred_cat[i], actual[i], y[i]])
    # print '\n'
    print z
    return render_template("op.html", op_val=y, pred=pred_cat, actual=actual, data=z)


if __name__ == "__main__":
    app.run(host='0.0.0.0')

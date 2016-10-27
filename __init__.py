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
    # pred_cat = [b.argmax() for b in pred]
    z = []
    for i in range(0, len(y)):
        z.append([pred[i], actual[i], y[i]])
    print z
    return render_template("op.html", data=z)


if __name__ == "__main__":
    app.run(host='0.0.0.0')

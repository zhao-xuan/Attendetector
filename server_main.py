from flask import Flask, escape, request, render_template

app = Flask(__name__, static_url_path='/static')
result = ''

@app.route('/update_status', methods = ['POST'])
def update_status():
    global result
    result = request.form.get("result")
    return ''

@app.route('/get_status')
def get_status():
    return result

if __name__ == '__main__':
	app.run(debug = True, host="0.0.0.0")
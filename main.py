from flask import Flask

app = Flask(
    __name__,
    template_folder = 'template/',
    static_folder = 'static/'
)

@app.route('/')
def index():
    return "Hello World!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
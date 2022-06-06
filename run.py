import os
from flask import Flask
from app.routes.index import routes


application = Flask(__name__)
application.static_folder = 'app/static'
application.template_folder = 'app/templates'
application.register_blueprint(routes)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'app', 'static', 'upload')
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if __name__ == '__main__':
    application.run(debug=True)

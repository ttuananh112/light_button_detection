from libs.connection.server import *
import libs.configs.conn as conn

if __name__ == "__main__":
    # start flask app
    app.run(host=conn.IP, port=conn.PORT)

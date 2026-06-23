from http.server import BaseHTTPRequestHandler
import json
import os
import joblib
import numpy as np

BASE = os.path.dirname(__file__)
modelo = joblib.load(os.path.join(BASE, "modelo_web.joblib"))
with open(os.path.join(BASE, "datos.json")) as f:
    DATOS = json.load(f)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or "{}")

        persona = dict(DATOS["medianas"])
        for k, v in body.items():
            if k in persona:
                persona[k] = float(v)

        fila = [persona[c] for c in DATOS["columnas"]]
        X = np.array([fila], dtype=float)
        prob = float(modelo.predict_proba(X)[0, 1])

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        salida = {"probabilidad": prob, "fuma": prob >= DATOS["umbral"]}
        self.wfile.write(json.dumps(salida).encode())

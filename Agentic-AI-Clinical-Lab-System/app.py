from flask import Flask, render_template, request, jsonify, send_file
import json
import os
from AIAgent import process_json_input

app = Flask(__name__)


OUTBOX_FILE = "outbox.json"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        patient_data = None

        # Option 1: uploaded JSON file
        if "file" in request.files and request.files["file"].filename:
            uploaded_file = request.files["file"]
            patient_data = json.load(uploaded_file)

        # Option 2: pasted JSON text
        elif request.form.get("json_input"):
            patient_data = json.loads(request.form.get("json_input"))

        else:
            return jsonify({"error": "Please upload or paste JSON lab data."}), 400

        # Support either one patient object or list of patients
        if isinstance(patient_data, list):
            results = [process_json_input(patient) for patient in patient_data]
        else:
            results = [process_json_input(patient_data)]

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download", methods=["GET"])
def download_outbox():
    if not os.path.exists(OUTBOX_FILE):
        return jsonify({"error": "outbox.json does not exist yet."}), 404

    return send_file(
        OUTBOX_FILE,
        as_attachment=True,
        download_name="outbox.json",
        mimetype="application/json"
    )


if __name__ == "__main__":
    app.run(debug=True)
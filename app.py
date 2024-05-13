from flask import Flask, render_template, request
from Final_Transformer import Summarizer

app = Flask(__name__)

transformer = Summarizer("P:/Checkpoints/point1499.pth")

# create an endpoint so we can render the index.html
@app.route('/')
def home():
    return render_template('index.html')

# create another end-point
@app.route('/information-extraction', methods=["GET", "POST"])
def summarize():
    if request.method == "POST":
        input_text = request.form["inputtext_"]  # Use square brackets for accessing form data, not parentheses
        processed_text = transformer.predict(input_text)  # Save the prediction result
        return render_template('results.html', summary=processed_text)
    return render_template('index.html')  # Render index.html for GET requests

# main function that runs the project
if __name__ == '__main__':
    app.run(debug=True, host="192.168.0.4", port=5000)

#  Create Run point
@app.route('/')
def home():
    return render_template('index.html')
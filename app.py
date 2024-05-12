from flask import Flask, render_template, request
import torch

# model_name = "articles_Field.pt"
# model_name2 = "highlights_Field.pt"

# create an object of flask and assign to "app" variable
app = Flask(__name__) # to access all routes through "app"
# transformer = Transformer
# transformer.loadModel()
# replace "user_input" with the value of the "name" attribute in the index.html input field in the form
# output=transformer.predict("user_input")

# def processing(text):
#     return text

def loadModel():
    # model_path = "Vocab/articles_Field.pt"
    # model = torch.load(model_path) # put the path name
    return torch.load("Vocab/articles_Field.pt")

# loadModel("articles_Field.pt") # torch.load('') # put the path name when calling the loadModel()
# loadModel()

'''
1. declare the name of the NLP model by model_name=Information_Extractor>
2. declare the device such as: device = "cuda" if torch.cuda.is_available() else "cpu"
3. model

# we could've used a pre-trained text summarizer transformer such as Pegasus
'''

# create an endpoint so we can render the index.html
@app.route('/')
def home():
    return render_template('index.html')

# create another end-point
@app.route('/information-extraction', methods=["GET", "POST"]) # declare the method as a list rather than normal string
def summarize():
    if request.method == "POST":
        # input_text = request.form("inputtext_")
        # input_text = "summarize" + inputText
        processed_text = "This is a placeholder summary generated by the transformer model."
        # print(inputText)
        # processed_text = generate_summary(input_text, articles_field, highlights_field, model, device)
        # return render_template('results.html', summary=processed_text)
        return render_template('results.html')
    return render_template('results.html')

# main function that runs the project
if __name__ == '__main__':
    app.run(debug=True)

# 1. Need to install the ThyTransformer.py
# -- Can do pip install on Terminal while being on the same working directory as the project
# -- 
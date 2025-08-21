import base64
from io import BytesIO
from PIL import Image
from transformers import pipeline
import nltk
import re
import string
import pickle
import sklearn
def hindi_tokenizer(text):
    tokens = nltk.word_tokenize(text,language='hindi',preserve_line=True)

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))

    words = [re_punc.sub('',w) for w in tokens]
    return words

textpipe = pipeline("text-classification", model="skandavivek2/spam-classifier")
imagepipe = pipeline("image-classification", model="Wvolf/ViT_Deepfake_Detection")

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
    
with open('svm.pkl','rb') as f:
    svm = pickle.load(f)
    
def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    input_type = job_input["type"]
    lang = job_input.get("lang", "en") 
    print(f"Received job with input: {job_input}")
    
    if input_type == "text":
        if lang == "hi":
            txt_vect = tfidf.transform([job_input["text"]])
            prediction = svm.predict(txt_vect)
            if prediction == 1 :
                return "SPAM"
            else:
                return "HAM"  
        else:
            return textpipe(job_input["text"])
    elif input_type == "image":
        image_data = base64.b64decode(job_input["image"])
        image = Image.open(BytesIO(image_data))
        return imagepipe(image)
    else:
        return {"error": "Invalid input type"}

import runpod
runpod.serverless.start({"handler": handler})

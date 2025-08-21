from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoImageProcessor, AutoModelForImageClassification

def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise

def get_text_classification_pipeline():
    '''
    Fetches the spam classifier pipeline from HuggingFace model hub.
    '''
    text_model_name = "skandavivek2/spam-classifier"
    
    text_model = fetch_pretrained_model(AutoModelForSequenceClassification, text_model_name)
    text_tokenizer = fetch_pretrained_model(AutoTokenizer, text_model_name)

    
    # Create pipeline for text classification
    classifier = pipeline("text-classification", model=text_model, tokenizer=text_tokenizer)
    image_model_name = "Wvolf/ViT_Deepfake_Detection"\
    
    image_model = fetch_pretrained_model(AutoModelForImageClassification, image_model_name)
    image_processor = fetch_pretrained_model(AutoImageProcessor, image_model_name)

    # Create pipeline for image classification

    image_classifier = pipeline("image-classification", model=image_model, image_processor=image_processor)

    print("Loaded Text Classification Pipeline")

    return classifier, image_classifier

if __name__ == "__main__":
    pipe = get_text_classification_pipeline()

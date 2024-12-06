from transformers import BartForConditionalGeneration


def load_model(model_name="facebook/bart-large-cnn"):
    """
    Загружает предварительно обученную модель BART.
    """
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model

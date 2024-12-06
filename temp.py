from transformers import BartTokenizer

# Путь к модели
model_path = "/run/media/wexel/7c741b0e-25f0-4df9-b7ae-24f423064507/home/glooma/Code/Python/ML/NLP_summirize_text/models/bart_pretrained"

# Если модель и токенизатор ещё не сохранены, загрузим их с HuggingFace
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
tokenizer.save_pretrained(model_path)  # Сохраним в нужной папке

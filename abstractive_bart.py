from transformers import BartForConditionalGeneration, BartTokenizer

# Путь к обученной модели
model_path = "/run/media/wexel/7c741b0e-25f0-4df9-b7ae-24f423064507/home/glooma/Code/Python/ML/NLP_summirize_text/models/bart_pretrained"

# Загрузка модели и токенизатора
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)


def abstractive_bart_summary(text, max_length=150, min_length=25):
    """
    Генерация краткого содержания текста с использованием Bart.

    Параметры:
    - text: Исходный текст для суммаризации.
    - max_length: Максимальная длина итогового текста.
    - min_length: Минимальная длина итогового текста.

    Возвращает:
    - Суммаризованный текст.
    """
    # Токенизация входного текста
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=1024)

    # Генерация резюме с использованием обученной модели
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,  # Для улучшения качества генерации
        length_penalty=2.0,  # Уменьшает вероятность слишком длинного вывода
        early_stopping=True  # Останавливает генерацию, если достигнут конец последовательности
    )

    # Декодирование результата в текст
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

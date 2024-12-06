import streamlit as st
from abstractive import abstractive_summary
from extractive import extractive_summary
from abstractive_bart import abstractive_bart_summary
from data_preprocessing import preprocess_text
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score


def run_interface():
    """
    Интерфейс для выбора модели суммаризации, настройки параметров и отображения метрик точности.
    """
    st.title("Автоматическое суммирование текста")

    # Ввод текста
    input_text = st.text_area("Введите текст для суммаризации:", height=200)

    # Выбор модели суммаризации
    model_type = st.selectbox(
        "Выберите модель суммаризации:",
        ("Экстрактивная", "T5-small", "BART-large"))

    # Настройка параметров
    if model_type == "T5-small" or model_type == "BART-large":
        min_length = st.slider("Минимальная длина резюме:", 10, 100, 25)
        max_length = st.slider("Максимальная длина резюме:", 50, 300, 100)
    else:
        num_sentences = st.slider("Количество предложений в резюме:", 1, 10, 3)

    # Кнопка для выполнения суммаризации
    if st.button("Сгенерировать резюме"):
        if not input_text.strip():
            st.warning("Пожалуйста, введите текст для суммаризации!")
            return

        # Отображение загрузки
        with st.spinner("Суммаризация текста... Пожалуйста, подождите."):
            # Предобработка текста
            sentences = preprocess_text(input_text)

            # Выполнение суммаризации
            if model_type == "Экстрактивная":
                summary = extractive_summary(
                    sentences, n_sentences=num_sentences)
            elif model_type == "T5-small":
                summary = abstractive_summary(
                    input_text, min_length=min_length, max_length=max_length)
            elif model_type == "BART-large":
                summary = abstractive_bart_summary(
                    input_text, min_length=min_length, max_length=max_length)

        st.success("Суммаризация завершена!")

        if model_type == "Экстрактивная":
            summary = " ".join(summary)

        # Отображение результата
        st.subheader("Резюме:")
        st.write(summary)

        # Расчёт метрик
        with st.spinner("Рассчитываем метрики точности..."):
            rouge_scores = calculate_rouge(input_text, summary)
            bleu = calculate_bleu(input_text, summary)
            bert = calculate_bertscore(input_text, summary)
            meteor = calculate_meteor(input_text, summary)

        # Отображение качества
        # Качество и индикатор
        if model_type == "Экстрактивная":
            quality = (rouge_scores['rouge1'].fmeasure + bleu) / 2
        else:
            quality = (bert['f1'] + meteor) / 2

        st.markdown("### Индикатор качества:")
        display_quality_bar(quality)

        # Отображение метрик
        st.subheader("Результаты метрик:")
        st.write(f"**ROUGE:** {rouge_scores}")
        st.write(f"**BLEU:** {bleu:.2f}")
        st.write(f"**BERTScore:** Precision: {bert['precision']:.2f}, Recall: {
                 bert['recall']:.2f}, F1: {bert['f1']:.2f}")
        st.write(f"**METEOR:** {meteor:.2f}")


def display_quality_bar(quality):
    """
    Отображает цветной бар в зависимости от качества.
    """
    # Нормализация качества от 0 до 1
    quality = max(0, min(quality, 1))

    # Генерация цвета: зеленый к красному
    red = int((1 - quality) * 255)
    green = int(quality * 255)
    color = f"rgb({red}, {green}, 0)"

    # HTML-код для полосы
    bar_html = f"""
    <div style="width: 100%; background-color: lightgray; border-radius: 5px; padding: 3px;">
        <div style="width: {quality * 100}%; background-color: {color}; height: 20px; border-radius: 5px;">
        </div>
    </div>
    """

    st.markdown(bar_html, unsafe_allow_html=True)


# Отдельные функции для метрик:


def calculate_rouge(reference, summary):
    """
    Рассчитывает ROUGE метрики для резюме.
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores


def calculate_bleu(reference, summary):
    """Рассчитывает BLEU метрику для резюме."""
    reference_tokens = reference.split()
    summary_tokens = summary.split()
    return sentence_bleu([reference_tokens], summary_tokens)


def calculate_bertscore(reference, summary):
    """Рассчитывает метрику BERTScore для резюме."""
    P, R, F1 = bert_score([summary], [reference], lang='ru')
    return {'precision': P[0].item(), 'recall': R[0].item(), 'f1': F1[0].item()}


def calculate_meteor(reference, summary):
    """Рассчитывает метрику METEOR для резюме."""
    reference_tokens = word_tokenize(reference)
    summary_tokens = word_tokenize(summary)
    return meteor_score([reference_tokens], summary_tokens)

from data_preprocessing import preprocess_text
from extractive import extractive_summary
from abstractive import abstractive_summary
import interface


def main():
    # Демонстрация работы
    '''
    text = """
    Burning Man - это недельное масштабное мероприятие в пустыне, посвященное "общине, искусству, самовыражению и самообеспечению", которое проводится ежегодно на западе США. [1][2] Название события происходит от его кульминационной церемонии: символического сжигания большого деревянного effigy, называемого как Человек, который происходит в предпоследнюю ночь, в субботу вечером перед День труда. [3] С 1990 года, событие было в Black Rock City на северо-западе Невады, временный город возведен в пустыне Black Rock около 100 миль (160 км) к северу-северо-востоку от Рено. Согласно основателю Burning Man Ларри Харви в 2004 году, это мероприятие руководствуется десятью установленными принципами: радикальное включение, дарование, декоммодификация, радикальная самоуверенность, радикальное самовыражение, коллективные усилия, гражданская ответственность, отсутствие следов, участие и непосредственность.
    """

    # Предобработка текста
    processed_text = preprocess_text(text)
    print("Preprocessed Text:", processed_text[:3])  # Первые 3 предложения

    # Экстрактивная суммаризация
    extractive_result = extractive_summary(processed_text)
    print("Extractive Summary:", extractive_result)

    # Абстрактивная суммаризация
    abstractive_result = abstractive_summary(" ".join(processed_text[:10]))
    print("Abstractive Summary:", abstractive_result)

    # Оценка
    reference = "Insert reference summary here."
    scores = evaluate_summary(reference, abstractive_result)
    print("Evaluation Scores:", scores)
    '''

    # Интерфейс
    interface.run_interface()


if __name__ == "__main__":
    main()

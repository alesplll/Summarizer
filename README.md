
# NLP Text Summarization Project

This project implements **text summarization** using both **abstractive** and **extractive** approaches. It leverages pre-trained models like **BART-large** and **T5-small** for abstractive summarization, while also providing a simple extractive summarization method. The models are fine-tuned on the [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail). 

---
## Features
- **Extractive Summarization**: Selects the most relevant sentences from the original text.
- **Abstractive Summarization**: Generates summaries using:
  - [BART-large](https://huggingface.co/facebook/bart-large)
  - [T5-small](https://huggingface.co/google-t5/t5-small)
- **Fine-tuning**: Easily fine-tune the model BART on custom datasets.
- **Custom Input Summarization**: Generate summaries for any input text.

---
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alesplll/Summarizer.git
   cd Summarizer
```

2. Set up a virtual environment and install dependencies:
    ```bash
python3.9 -m venv myenv
source myenv/bin/activate  # Linux/MacOS
myenv\Scripts\activate     # Windows
pip install -r requirements.txt
    ```

---
## Usage

### Start the Web-Interface
```bash
source myenv/bin/activate  # Linux/MacOS
myenv\Scripts\activate     # Windows
```

```bash
streamlit run main.py
```
### Training the Models
1. **Prepare the dataset**:  
    Adjust dataset size and parameters in `data/preprocess.py` if needed.
2. **Train the models**:  
    Run the training script for the desired model. For example:
    ```bash
    python train.py
    ```
3. Fine-tuned models will be saved in the `models/` directory.

---
## File Structure

```

Summarizer/ 
├── data/ 
│   └── preprocess.py              # Data preparation script settings
├── models/ 
│   ├── bart_pretrained/           # Fine-tuned BART model 
├── train.py                       # Training script 
├── extractive.py                  # Script for extractive summarization 
├── abstractive.py                 # T5-small summarization 
├── abstractive_bart.py            # BART-large summarization 
├── data_preprocessing.py          # Additional preprocessing for extractive
├── interface.py                   # Script for user web-interface 
├── main.py                        # Entry point for the project 
├── requirements.txt               # Python dependencies 
└── README.md                      # Project documentation
```

---
## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing pre-trained models and tokenizers.
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail) for summarization data.
- [NLTK](https://www.nltk.org/) for extractive summarization tools.

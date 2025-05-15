#pip install torch transformers sumy googletrans==4.0.0-rc1
#pip install datasets
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from googletrans import Translator
import nltk
nltk.download('punkt_tab')

# Load T5-small model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def translate_text(text, src_lang, dest_lang):
    translator = Translator()
    translated_text = translator.translate(text, src=src_lang, dest=dest_lang)
    return translated_text.text

def abstractive_summary(text, max_length=150, min_length=50):
    """Generate abstractive summary using T5-small."""
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extractive_summary(text, num_sentences=3):
    """Generate extractive summary using LSA method."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return "".join(str(sentence) for sentence in summary)

def sumup(text, opl):
    print("Text input: $1", text)

    # Detect language
    detected_lang = Translator().detect(text).lang

    # Translate to English for processing
    text_en = translate_text(text, detected_lang, 'en')

    # Generate summaries
    abstractive = abstractive_summary(text_en)
    extractive = extractive_summary(text_en)

    output_lang = opl

    # Translate summaries to chosen language
    abstractive_translated = translate_text(abstractive, 'en', output_lang)
    extractive_translated = translate_text(extractive, 'en', output_lang)

    print("\n--- Summaries ---")
    print(f"Abstractive Summary ({output_lang}):", abstractive_translated)
    print(f"Extractive Summary ({output_lang}):", extractive_translated)
    dict = {}
    dict['abs_sum'] = abstractive_translated
    dict['ext_sum'] = extractive_translated
    return dict

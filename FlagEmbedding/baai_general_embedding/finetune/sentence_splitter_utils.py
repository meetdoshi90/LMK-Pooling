import pysbd
import spacy
from sentence_splitter import SentenceSplitter
import logging


logger = logging.getLogger(__name__)

# list of languages supported by pySBD
PYSBD_LANGS = {
    'en', 'hi', 'mr', 'zh', 'es',
    'am', 'ar', 'hy', 'bg', 'ur',
    'ru', 'pl', 'fa', 'nl', 'da',
    'fr', 'my', 'el', 'it', 'ja',
    'de', 'kk', 'sk'
}

class EnglishSentenceSplitter:
    def __init__(self):
        self.sentence_splitter = SentenceSplitter(language='en', timeout=60) # 1 minute timeout

    def split(self, text: str):
        try:
            splitted_text = self.sentence_splitter.split(text)
        except Exception as e:
            logger.info(f'Error {e}') # Timeout catch
            splitted_text = [text]
        return splitted_text
    
    def split_longtext(self, text: str):
        return self.split(text) # For English data we do not run into many regex timeouts beyond 8k MSL

class GeneralSentenceSplitter:
    def __init__(self):
        self._pysbd_segmenters = {
            lang: pysbd.Segmenter(language=lang, clean=False)
            for lang in PYSBD_LANGS
        }
        try:
            self._xx_nlp = spacy.load("xx_sent_ud_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'xx_sent_ud_sm' not found. "
                "Please install it via `python -m spacy download xx_sent_ud_sm`."
            ) from e

    def split(self, text: str, lang: str = None):
        """
        Split text into sentences
        """
        use_lang = lang if lang is not None else 'en'

        if use_lang in self._pysbd_segmenters:
            seg = self._pysbd_segmenters[use_lang]
            try:
                return seg.segment(text)
            except Exception:
                pass
        with self._xx_nlp.memory_zone():
            doc = self._xx_nlp(text)
            sents = [sent.text for sent in doc.sents]
        return sents

    def split_longtext(self, text: str, lang: str = None, chunk_size: int = 8000): # increasing chunk size since we already have a 5 sec timeout
        use_lang = lang if lang is not None else 'en'

        if len(text) <= chunk_size:
            if use_lang in self._pysbd_segmenters:
                seg = self._pysbd_segmenters[use_lang]
                try:
                    return seg.segment(text)
                except Exception:
                    pass
            with self._xx_nlp.memory_zone():
                doc = self._xx_nlp(text)
                sents = [sent.text for sent in doc.sents]
            return sents

        sents = []
        offset = 0
        text_len = len(text)
        while offset < text_len:
            chunk = text[offset : offset + chunk_size]
            offset += chunk_size

            if use_lang in self._pysbd_segmenters:
                seg = self._pysbd_segmenters[use_lang]
                try:
                    chunk_sents = seg.segment(chunk)
                except Exception: # 5 sec timer exception
                    with self._xx_nlp.memory_zone():
                        doc = self._xx_nlp(chunk)
                        chunk_sents = [sent.text for sent in doc.sents]
            else:
                with self._xx_nlp.memory_zone():
                    doc = self._xx_nlp(chunk)
                    chunk_sents = [sent.text for sent in doc.sents]

            sents.extend(chunk_sents)

        return sents


# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bundestag Topic Modelling
# Eyeballing parliamentary minutes for two election periods using topic modelling is performed in this notebook. The initial creation of n-grams may take a long time!
# - Language model from spaCy
# - Corpus and LDA with gensim
# - Visualization with pyLDAvis

# %% [markdown]
# ## Setup
# %%
import codecs
import time
from pathlib import Path

import _pickle as pickle
import pyLDAvis
import pyLDAvis.gensim
import spacy
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Phrases
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import LineSentence
from IPython.core.display import HTML, display

# %%
# Use the full browser window width
display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
# Create needed subfolder in 'data' and subfolder 'out'
Path("data/tm").mkdir(parents=True, exist_ok=True)
Path("out").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# #### Configuration
# - Models are large in size and quick to compute -> save only when re-running the notebook frequently
# - SpaCy provides a small and a large language model for German
# - Notebook 02 produces two separate speech text files, `..._a.txt` contains speech fragments per line (i.e. per document when processed here), `..._b.txt` contains whole speeches per line
#
# The prefix encodes the configuration and is added to all file paths.

# %%
SAVE_MODELS = True
language_model = "de_core_news_lg"
# language_model = 'de_core_news_sm'
gerNLP = spacy.load(language_model)
# speeches_txt_filepath = 'data/plpr_alltext_a.txt'
speeches_txt_filepath = "data/plpr_alltext_b.txt"

# %%
prefix = f"{speeches_txt_filepath[-5:-4]}_{language_model[-2:]}_"  # text file option (a or b) and language model (small sm or large lg)
prefix


# %% [markdown]
# ### Helper Functions
# The helper functions help with text preprocessing, the creation of n-grams, and to show results. The creation of the lemmatized sentence corpuse is memory intense. Reduce batch size and the number of parallel processes if needed.

# %%
def preview(filepath, N=1000):
    """Previews N characters of file."""
    with open(filepath) as temp:
        head = next(temp)
        head = head[:N]
    temp.close()
    return head


def preview_lines(filepath, N=5):
    """Previews N lines of file."""
    with open(filepath) as temp:
        head = [next(temp) for i in range(N)]
    temp.close()
    return head


def punct_space(token):
    """Removes punctuation and whitespace."""
    return token.is_punct or token.is_space


def line_speech(filename):
    """Reads lines and ignores lines breaks"""
    with codecs.open(filename, encoding="utf_8") as f:
        for speech in f:
            yield speech.replace("\\n", "\n")


def lemmatized_sentence_corpus_to_file(input_file, output_file):
    """Parses speeches with spaCy, writes lemmatized sentences to file."""
    with codecs.open(output_file, "w", encoding="utf_8") as f:
        for parsed_speech in gerNLP.pipe(
            line_speech(input_file), batch_size=100, n_process=8
        ):
            for sent in parsed_speech.sents:
                parsed_sent = " ".join(
                    [token.lemma_ for token in sent if not punct_space(token)]
                )
                f.write(parsed_sent + "\n")


# %% [markdown]
# ## Unigrams
# The text file is a text-only extract of the speeches of the plenary protocols dataframe in `nb_02`. To create unigrams, all text is cleaned by stripping junk such as stop words or meaningless filter workds, conjugated words are reveresed to their base form.

# %%
preview(speeches_txt_filepath)

# %%
# %%time
# long running
unigram_sentences_filepath = f"data/tm/{prefix}unigram_sent_all.txt"

if Path(unigram_sentences_filepath).exists():
    print(f"Unigram sentences available at {unigram_sentences_filepath}")
else:
    print(f"Unigram sentences not available. Now creating {unigram_sentences_filepath}")
    lemmatized_sentence_corpus_to_file(
        input_file=speeches_txt_filepath, output_file=unigram_sentences_filepath
    )

unigram_sentences = LineSentence(unigram_sentences_filepath)

preview_lines(unigram_sentences_filepath)

# %% [markdown]
# ## Bigrams
# Bigrams (or any larger structure of n-grams) represent word pairs (or triplets, quadruples, etc.) of words commonly appearing together. "Renewable" and "energy" used independetly do not convey the same meaning as when being used together in "renewable_energy".

# %%
# %%time
bigram_model_filepath = f"data/tm/{prefix}bigram_model_all"
if Path(bigram_model_filepath).exists():
    print(f"Bigram model available at {bigram_model_filepath}")
    bigram_model = Phrases.load(bigram_model_filepath)
else:
    print(f"Bigram model not available. Now creating {bigram_model_filepath}")
    bigram_model = Phrases(unigram_sentences)
    if SAVE_MODELS:
        bigram_model.save(bigram_model_filepath)

# %%
# %%time
# short running
bigram_sentences_filepath = f"data/tm/{prefix}bigram_sent_all.txt"
if Path(bigram_sentences_filepath).exists():
    print(f"Bigram sentences available at {bigram_sentences_filepath}")
else:
    print(f"Bigram sentences not available. Now creating {bigram_sentences_filepath}")
    with codecs.open(bigram_sentences_filepath, "w", encoding="utf_8") as f:
        for unigram_sentence in unigram_sentences:
            bigram_sentence = " ".join(bigram_model[unigram_sentence])
            f.write(bigram_sentence + "\n")

bigram_sentences = LineSentence(bigram_sentences_filepath)

preview_lines(bigram_sentences_filepath)

# %% [markdown]
# ## Trigrams
# As bigrams are used to create trigrams, there is the chance of two bigrams being combined which would be a 4-gram.

# %%
# %%time
trigram_model_filepath = f"data/tm/{prefix}trigram_model_all"
if Path(trigram_model_filepath).exists():
    print(f"Trigram model available at {trigram_model_filepath}")
    trigram_model = Phrases.load(trigram_model_filepath)
else:
    print(f"Trigram model not available. Now creating {trigram_model_filepath}")
    trigram_model = Phrases(bigram_sentences)
    if SAVE_MODELS:
        trigram_model.save(trigram_model_filepath)

# %%
# %%time
# short running
trigram_sentences_filepath = f"data/tm/{prefix}trigram_sent_all.txt"
if Path(trigram_sentences_filepath).exists():
    print(f"Trigram sentences available at {trigram_sentences_filepath}")
else:
    print(f"Trigram sentences not available. Now creating {trigram_sentences_filepath}")
    with codecs.open(trigram_sentences_filepath, "w", encoding="utf_8") as f:
        for bigram_sentence in bigram_sentences:
            trigram_sentence = " ".join(trigram_model[bigram_sentence])
            f.write(trigram_sentence + "\n")

trigram_sentences = LineSentence(trigram_sentences_filepath)

preview_lines(trigram_sentences_filepath)

# %%
# %%time
# long running
trigram_speeches_filepath = f"data/tm/{prefix}trigram_transformed_speeches_all.txt"
if Path(trigram_speeches_filepath).exists():
    print(f"Trigram speeches available at {trigram_speeches_filepath}")
else:
    print(f"Trigram speeches not available. Now creating {trigram_speeches_filepath}")
    with codecs.open(trigram_speeches_filepath, "w", encoding="utf_8") as f:
        for parsed_speech in gerNLP.pipe(
            line_speech(speeches_txt_filepath), batch_size=100, n_process=15
        ):

            # lemmatize the text, removing punctuation and whitespace
            unigram_speech = [
                token.lemma_ for token in parsed_speech if not punct_space(token)
            ]

            # apply the first-order and second-order phrase models
            bigram_speech = bigram_model[unigram_speech]
            trigram_speech = trigram_model[bigram_speech]

            # remove any remaining stopwords
            trigram_speech = [
                term
                for term in trigram_speech
                if term.lower() not in spacy.lang.de.STOP_WORDS
            ]
            # stop words found here: https://github.com/explosion/spaCy/blob/master/spacy/lang/de/stop_words.py

            # write the transformed speech as a line in the new file
            trigram_speech = " ".join(trigram_speech)
            f.write(trigram_speech + "\n")

preview_lines(trigram_speeches_filepath, N=2)

# %%
print(
    f"File {trigram_speeches_filepath} contains {sum(1 for line in open(trigram_speeches_filepath))} documents"
)

# %% [markdown]
# ## Latent Dirichlet Allocation
# In this section, the text is transformed into a corpus, which is the collection of documents over which topics are discovered using LDA. As a first intermediate step, the speech documents are represented with a dictionary, where n-grams are keys and occurances counts within speech documents are the respective values.

# %% [markdown]
# The parameters `THRES_BELOW` AND `THRES_ABOVE` define, which keywords can define a topic. `THRES_BELOW` is the minimum number of documents, in which a keyword needs to occur to be able to define a topic. `THRES_ABOVE` is a relative value, it defines the maximum fraction of documents, which may contain a keyword for the keyword to be able to define topics. Accordingly, keywords which are too common, cannot define a topic, and special terminology of a single speech does not, either.

# %%
# %%time
# short running
THRESH_BELOW = 2
THRESH_ABOVE = 0.01
thres_suffix = f"TB{str(THRESH_BELOW)}_TA{str(THRESH_ABOVE)}".replace(".", "")
trigram_dictionary_filepath = f"data/tm/{prefix}trigram_dict_all_{thres_suffix}.dict"
if Path(trigram_dictionary_filepath).exists():
    print(f"Trigram dictionary available at {trigram_dictionary_filepath}")
    trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)
else:
    print(
        f"Trigram dictionary not available. Now creating {trigram_dictionary_filepath}"
    )

    trigram_speeches = LineSentence(trigram_speeches_filepath)

    trigram_dictionary = Dictionary(trigram_speeches)

    # filter tokens that are very rare or too common from
    # the dictionary (filter_extremes) and reassign integer ids (compactify)
    trigram_dictionary.filter_extremes(no_below=THRESH_BELOW, no_above=THRESH_ABOVE)
    trigram_dictionary.compactify()

    trigram_dictionary.save(trigram_dictionary_filepath)


# %% code_folding=[0]
def trigram_bow_generator(filepath):
    """
    generator function to read speeches from a file
    and yield a bag-of-words representation
    """

    for speech in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(speech)


# %%
# %%time
# short running
trigram_bow_filepath = f"data/tm/{prefix}trigram_bow_corpus_all_{thres_suffix}.mm"
if Path(trigram_bow_filepath).exists():
    print(f"Trigram bag-of-words available at {trigram_bow_filepath}")
else:
    print(f"Trigram bag-of-words not available. Now creating {trigram_bow_filepath}")
    # generate bag-of-words representations for
    # all speeches and save them as a matrix
    MmCorpus.serialize(
        trigram_bow_filepath, trigram_bow_generator(trigram_speeches_filepath)
    )

# load the finished bag-of-words corpus from disk
trigram_bow_corpus = MmCorpus(trigram_bow_filepath)

# %% [markdown]
# ## Topic Models & Visuals
# Latent topics are finally within the corpus are finally derived. "Latent" means that topic belongingness may not be obvious at first sight for a document. The output is by no means finite and requires manual review and validation.

# %%
# %%time
# medium-long running
topics = [5, 10, 15, 30, 50, 100, 250, 500]
for number_of_topics in topics:
    # topic model
    lda_model_filepath = (
        f"data/tm/{prefix}lda_model_{thres_suffix}_{str(number_of_topics)}"
    )
    if Path(lda_model_filepath).exists():
        print(f"Trigram bag-of-words available at {lda_model_filepath}")
        # load the finished LDA model from disk
        lda = LdaMulticore.load(lda_model_filepath)
    else:
        print(f"Trigram bag-of-words not available. Now creating {lda_model_filepath}")

        lda = LdaMulticore(
            trigram_bow_corpus,
            num_topics=number_of_topics,
            id2word=trigram_dictionary,
            workers=8,
        )
        lda.save(lda_model_filepath)

    # topic model visual
    LDAvis_data_filepath = (
        f"data/tm/{prefix}ldavis_prepared_{thres_suffix}_{str(number_of_topics)}"
    )
    if Path(LDAvis_data_filepath).exists():
        print(f"LDA Visualization available at {LDAvis_data_filepath}")
        with open(LDAvis_data_filepath, "rb") as f:
            LDAvis_prepared = pickle.load(f)
    else:
        print(f"LDA visualization not available. Now creating {LDAvis_data_filepath}")
        LDAvis_prepared = pyLDAvis.gensim.prepare(
            lda, trigram_bow_corpus, trigram_dictionary
        )

        with open(LDAvis_data_filepath, "wb") as f:
            pickle.dump(LDAvis_prepared, f)

    # topic model html visual
    LDAvis_html_filepath = (
        f"out/{prefix}lda_viz_{thres_suffix}_{str(number_of_topics)}.html"
    )
    if Path(LDAvis_html_filepath).exists():
        print(f"LDA Visualization available at {LDAvis_html_filepath}")
    else:
        print(
            f"LDA html visualization not available. Now creating {LDAvis_html_filepath}"
        )
        pyLDAvis.save_html(LDAvis_prepared, LDAvis_html_filepath)

# %% [markdown]
# ## Single model review

# %%
DEFAULT_NO_TOPICS = 250
DEFAULT_THRESHS = "TB2_TA005"
lda_model_filepath = (
    f"data/tm/{prefix}lda_model_{DEFAULT_THRESHS}_{str(DEFAULT_NO_TOPICS)}"
)
# load the finished LDA model from disk
lda = LdaMulticore.load(lda_model_filepath)
LDAvis_data_filepath = (
    f"data/tm/{prefix}ldavis_prepared_{thres_suffix}_{str(DEFAULT_NO_TOPICS)}"
)
with open(LDAvis_data_filepath, "rb") as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.display(LDAvis_prepared)


# %% [markdown]
# On Github, the notebook is statically rendered, the LDA visualization is interactive. View with nbviewer instead: [Jupyter nbviewer](https://nbviewer.jupyter.org/github/sebas-seck/bundestag_nlp/blob/main/nb_03_topic_modelling.ipynb#topic=0&lambda=1&term=)

# %%
def lda_description(review_text, min_topic_freq=0.05):
    """
    accept the original text of a review and (1) parse it with spaCy,
    (2) apply text pre-proccessing steps, (3) create a bag-of-words
    representation, (4) create an LDA representation, and
    (5) print a sorted list of the top topics in the LDA representation
    """

    # parse the review text with spaCy
    parsed_review = gerNLP(review_text)

    # lemmatize the text and remove punctuation and whitespace
    unigram_review = [token.lemma_ for token in parsed_review if not punct_space(token)]

    # apply the first-order and secord-order phrase models
    bigram_review = bigram_model[unigram_review]
    trigram_review = trigram_model[bigram_review]

    # remove any remaining stopwords
    trigram_review = [
        term for term in trigram_review if not term in spacy.lang.de.STOP_WORDS
    ]

    # create a bag-of-words representation
    review_bow = trigram_dictionary.doc2bow(trigram_review)

    # create an LDA representation
    review_lda = lda[review_bow]

    # sort with the most highly related topics first
    review_lda = sorted(review_lda, key=lambda topic_number_freq: -topic_number_freq[1])

    for topic_number, freq in review_lda:
        if freq < min_topic_freq:
            break

        # print the most highly related topic names and frequencies
        print("{:25} {}".format(topic_number, round(freq, 3)))


# %% [markdown]
# ## Speech Review

# %%
review_text1 = """Herr Minister, möglicherweise ist das ein Anlass, um über andere Strukturen nachzudenken. Im Land Brandenburg, aus dem ich komme,
    gibt es im Süden einen Bestand von 60 000 Schweinen an einem Standort. Stellen wir uns vor, dass dieser Standort wegen der Afrikanischen
    Schweinepest auf einmal in einer Restriktionszone liegt. Dann werden wir wahrscheinlich nicht umhinkommen, den gesamten Bestand zu töten.
    Ist es nicht an der Zeit, einmal ernsthaft darüber nachzudenken, ob solche Megaställe nicht der Vergangenheit angehören sollten und ob unter
    Aspekten der Tierseuchenbekämpfung nicht Regionen mit sehr dichtem Tierbestand als auch solche riesengroßen Bestände vermieden werden sollten?
    Das ist einfach sehr schwierig in einer Tierseuchensituation zu händeln. Ich glaube zudem, dass die in Rede stehenden Maßnahmen ethisch nicht
    mehr vertretbar sind. Deswegen lautet meine Frage: Müssen wir nicht auch über Strukturen bei den Tierbeständen nachdenken?"""

# %%
review_text2 = """Vielen Dank, Herr Präsident. – Herr Kollege Kekeritz, gestern fand eine informelle Tagung der Entwicklungsminister der
    Europäischen Union statt. Auf der Tagesordnung stand unter anderem der mehrjährige Finanzrahmen der Europäischen Union. Die Kommission bereitet
    die Debatte vor. Das Europäische Parlament wie auch der Ministerrat in allen seinen Formationen wird sich zu der Frage positionieren müssen,
    wie der Haushalt der Europäischen Union im Zeitrahmen des nächsten mehrjährigen Finanzrahmens aufzustellen ist. In diesem Zusammenhang hat
    Entwicklungsminister Dr. Gerd Müller dazu aufgerufen, die internationalen Aufgaben der Europäischen Union, insbesondere mit Blick auf Afrika,
    deutlich zu stärken."""

# %%
review_text3 = """Welche konkreten rechtlichen Überlegungen haben die Ostbeauftragte Iris Gleicke und das Bundeswirtschaftsministerium dazu
    veranlasst, für eine Studie des Göttinger Instituts für Demokratieforschung zum Thema 'Rechtsextremismus und Fremdenfeindlichkeit in
    Ostdeutschland', die nach eigenen Angaben von Iris Gleicke selbst nach Nacherfüllungsmöglichkeit eine 'schlicht nicht hinnehmbare Schlamperei'
    darstellt, von der sie sich öffentlich distanziert hat und die für sie 'jeden Wert ... verloren' hatte, nicht nur die Rückforderung von
    bereits ausgezahlten Geldern zu unterlassen, sondern auch noch zu einem Zeitpunkt, als die Unbrauchbarkeit der Studie bereits bekannt war,
    einen bis dahin noch nicht ausgezahlten Betrag hierfür zu zahlen, wie unter anderem die Zeitung 'Die Welt' am 12. Februar 2018 berichtet hat,
    und wie hoch war der Betrag, der erst nach Bekanntwerden der Mangelhaftigkeit der Studie an das Göttinger Institut für Demokratieforschung bzw.
    die Georg-August-Universität Göttingen ausgezahlt wurde?"""

# %%
lda_description(review_text1)

# %%
lda_description(review_text2)

# %%
lda_description(review_text3)

# %%

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 01 spaCy Exploration

# +
import os

import pandas as pd
import spacy
from spacy import displacy

# -

# ### Helper Functions


def show(table):
    with pd.option_context("display.max_colwidth", None):
        display(table)


# ### Starting with spaCy

# Prints version of spaCy in use
print(spacy.__version__)

# The primary language model will be the larger model `de_core_news_lg`. To point out the differences, the small model `de_core_news_sm` is loaded, too.

gerNLPsm = spacy.load("de_core_news_sm")
gerNLPlg = spacy.load("de_core_news_lg")

# Calling 'spacy.info()' on the German model returns the model's meta data.
info = spacy.info("de_core_news_lg")

# # Linguistic Features in Speech Parsing

# For the exploration of the configuration and features of the library spaCy, a speech given on March 23rd 2011 during the parliament's question time by Ursula Heinen-Esser (CDU) is loaded as a document. At the time, Ursula Heinen-Esser was parliamentary undersecretary to the Federal Minister for the Environment, Nature Conservation and Nuclear Safety.
#
# To get started, the raw text is converted to a doc object using the German language model previously loaded. As the doc object is created, tokenization is done, too.

# +
doc_txt = open("data/sample_speech.txt", "r", encoding="utf8")
doc = gerNLPlg(doc_txt.read())

with pd.option_context("display.max_colwidth", 25):
    print(doc)
# -

sentences = [sentence.orth_ for sentence in doc.sents]
words = [token.orth_ for token in doc if token.pos_ != "PUNCT"]
print(
    "The sample speech contains "
    + str(len(sentences))
    + " sentences and "
    + str(len(words))
    + " words in total."
)

# ## Part-of-Speech (POS) tagging
# The next chunk of code returns a table of tokens 289 to 317, a sample sentence of the speech, with labels. spaCy tokenizes everything (words, numbers, punctuation, etc.) except single spaces next to words. The POS label indicate language-universal syntactic token positions, the TAG label indicates position labels specifically for the German language using the STTS. https://explosion.ai/blog/german-model#Data-sources

# +
pos_tags = []  # empty list to be filled later
# for every token the loop appends the tags and labels to the list
for token in doc:
    pos_tags.append(
        (token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.is_stop)
    )

# converts the list to a dataframe which can be visualized nicely in markdown
pos_tags = pd.DataFrame(
    pos_tags, columns=("TEXT", "LEMMA", "POS", "TAG", "DEP", "STOP")
)
show(pos_tags[289:317])
# -

# ### Decoding the tag labels
# Calling *spacy.explain()* on the TAG or POS will return a more profound explanation of the tag closer to human understanding of language.

# +
tags_explained = []
for token in doc:
    tags_explained.append(
        (
            token.text,
            token.pos_,
            spacy.explain(token.pos_),
            token.tag_,
            spacy.explain(token.tag_),
        )
    )

tags_explained = pd.DataFrame(
    tags_explained, columns=("TEXT", "POS", "POS_Explained", "TAG", "TAG_Explained")
)
show(tags_explained[289:317])
# -

# In the sample speech of 65 sentences and 1019 words in total there are 40 unique tags, each shown with an example in the 'TEXT' columns in the next table.
# \label{se-unique-tags}

unique_tags = tags_explained.drop_duplicates("TAG_Explained").sort_values(by=["TAG"])[
    ["TAG", "TAG_Explained", "POS", "POS_Explained", "TEXT"]
]
unique_tags = pd.DataFrame(
    unique_tags, columns=("TAG", "TAG_Explained", "POS", "POS_Explained", "TEXT")
).reset_index()
print(f"Table shaped {str(unique_tags.shape)} with unique values")
show(
    pd.DataFrame(
        unique_tags, columns=("TAG", "TAG_Explained", "POS", "POS_Explained", "TEXT")
    )
)

# ## Dependencies
# The German language exhibits a special morpological characteristic in comparison to English. German is non-projective and generally less restrictive, which allows for various word orders in a sentence without changing the meaning. It's recognizable by crossing dependency arcs. See this spaCy blog post on [word order](https://explosion.ai/blog/german-model#word-order) and this introcution to the [TIGER annotation treebank](https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/annotation/tiger_introduction.pdf) for more detail on German linguistics for NLP.
#
# With Displacy, another tool from the same creators of spaCy, visualizes parsed text with POS tags and the dependency tags between related tokens.

# +
displacy_options = {
    "fine_grained": False,
    "add_lemma": True,
    "collapse_phrases": False,
    "compact": True,
    "arrow_stroke": 1,
    "distance": 100,
}


def displacy_visual(
    spaCy_doc, style="dep", options=displacy_options, minify=True, jupyter=True
):
    """Takes a spaCy document and returns a visual of the dependency parse."""
    visual = displacy.render(spaCy_doc, style, options=options)
    return visual


# -

example_text = "Ich glaube, wir sollten bei dieser Branche einen Schwerpunkt setzen, da sie uns weg von \
Atomenergie und fossilen Energieträgern hin zu dezentralen Lösungen führt, und nicht schon jetzt Kürzungen \
vornehmen, obwohl die Branche noch nicht einmal richtig etabliert ist."
example = gerNLPlg(example_text)
example_sm = gerNLPsm(example_text)

# ### Parsed dependencies with the small model versus the large model
# Admittedly, the example at hand is a long, convoluted sentence. But it's a real-world example being part of the full document described above.
#
# The example starts with a subclause kicking-off indirect speech _"Ich glaube"_, followed by a main clause _"wir sollten bei dieser Branche einen Schwerpunkt setzen und nicht schon jetzt Kürzungen vornehmen"_ with an embedded subclause _"da sie uns weg von Atomenergie und fossilen Energieträgern hin zu dezentralen Lösungen führt"_, and rounded-off by another subclause _"obwohl die Branche noch nicht einmal richtig etabliert ist."_. The last clause is a subclause due to the subordinating conjunction `SCONJ`. The full example with explained dependency and POS tags follows later in the notebook.
#
# The first visual uses the small model, the second uses the large model. There is only minor difference to be spotted:
#
# - Parsed with the small model, "führt" is connected with "und" as a coordinating conjunction (edge tag `cd`)
# - Parsed with the large model, "und" is no longer connected with "führt", but with "setzen" in the first main clause as a coordinating conjunction
#
# The small model identifies the second part of the main clause to be a coordinating conjunction to the embedded subclause, though coordinating conjunctions only connect two main clauses. The large model connects the second part of the main clause with the first, which is right.
#
# This deviation into the differences between the language models only touches the surface. I'll keep using the larger model from now.

displacy_visual(example_sm)

# ### Parsed dependencies with large model

displacy_visual(example)

short_doc = gerNLPlg(
    "Die Ereignisse in Japan haben uns gezeigt, dass das sogenannte Restrisiko durchaus existent ist \
und dass es sich hierbei nicht nur um eine rechnerische Größe handelt."
)

displacy_visual(short_doc)

# ### Investigating the parse tree
# The arcs shown in the dependency parsing visualization with displacy define the syntactic relation between two words. The arcs are directional, which indicates the status of head and child inside of the dependency tree. In the following chunk, the entire dependency tree is parsed.

# +
parse_tree = []
for token in example:
    parse_tree.append(
        (
            token.text,
            token.pos_,
            spacy.explain(token.pos_),
            token.dep_,
            spacy.explain(token.dep_),
            token.head.text,
            token.head.pos_,
            [child for child in token.children],
        )
    )

parse_tree = pd.DataFrame(
    parse_tree,
    columns=(
        "TOKEN_TEXT",
        "TEXT_POS",
        "POS_Explained",
        "DEP",
        "DEP_Explained",
        "HEAD_TEXT",
        "HEAD_POS",
        "CHILDREN",
    ),
)
show(parse_tree)

# +
parse_tree = []
for token in doc:
    parse_tree.append(
        (
            token.text,
            token.pos_,
            spacy.explain(token.pos_),
            token.dep_,
            spacy.explain(token.dep_),
            token.head.text,
            token.head.pos_,
            [child for child in token.children],
        )
    )

parse_tree = pd.DataFrame(
    parse_tree,
    columns=(
        "TOKEN_TEXT",
        "TEXT_POS",
        "POS_Explained",
        "DEP",
        "DEP_Explained",
        "HEAD_TEXT",
        "HEAD_POS",
        "CHILDREN",
    ),
)
show(parse_tree[289:317])
# -

# The parse tree above shows that dependency tags are still accessed via tokens, despite actually explaining what the relationship between two tokens looks like. The dependency between a token and its head can be accessed via the token. As the sentence root does not have a head, no dependency can be accessed via the root.

unique_dep = parse_tree.drop_duplicates("DEP").sort_values(by=["DEP"])[
    ["DEP", "DEP_Explained"]
]
show(unique_dep)

# Viewing the table it becomes clear that first degree relations can proove to be meaningful. For example, 'Restrisiko' (remaining risk) is the direct head to the adjective 'sogenannte' (so-called) which carries judging of the noun 'Restrisiko'.
# Nevertheless, working only with first degree relationships would disregard plenty of information hidden in the sentence. For example, the adjective 'durchaus' (indeed) which describes the adverb 'existent' (existant) is related to 'Restrisiko' through the auxiliary verb 'ist' (is). This valuable information is lost if the dependency tree is not crawled through in order to investigate local trees.

# ### Crawling through the local tree
# Previously we have only investigated single archs. Looking at local trees will put our focus on a per sentence level. The ultimate interest is to iterate over the token to find chains, in other words to follow an arch to another arch to another arch. Later, in the analysis, conditions for the continuation of the iteration can be set, such as the POS/TAG label having to be a verb and then an adverb since we are looking at such discriptive words.
#
# #### Local Surrounding per Token
# It is of interest to follow along the word hierarchy. Therefore, let's first look at the local surroundings of each token in the next table. Syntactic children are words which are connected by an arch to the token, the distinction in left and right refers to their appearance before or after the token. This number gives an idea about the surrounding of each token. The token head, which is also returned, will indicate the location of the root - it is where both token and token head are similar.

# +
local_trees = []
for token in doc:
    local_trees.append(
        (
            token.text,
            token.head,
            token.n_lefts + token.n_rights,
            token.n_lefts,
            token.n_rights,
        )
    )

local_trees = pd.DataFrame(
    local_trees,
    columns=("TOKEN_TEXT", "TOKEN_HEAD", "TOTAL_CHILD", "LEFT_CHILD", "RIGHT_CHILD"),
)
show(local_trees[289:317])
# -

# #### Descendants per Token
# This looks at the subtree of the token and returns any words the token archs out to. Thus, also words beyond the first degree are included.

# +
descendants = []
for token in example:
    descendants.append(
        (
            token.text,
            [descendant.text for descendant in token.subtree if token != descendant],
        )
    )

descendants = pd.DataFrame(descendants, columns=("TOKEN_TEXT", "DESCENDANTS"))
show(descendants)

# +
descendants = []
for token in short_doc:
    descendants.append(
        (
            token.text,
            [descendant.text for descendant in token.subtree if token != descendant],
        )
    )

descendants = pd.DataFrame(descendants, columns=("TOKEN_TEXT", "DESCENDANTS"))
show(descendants)
# -

# #### Anchestors per token
# This looks at the ancestors of the token and returns any words which archs out to the token.

# +
ancestor = []
for token in short_doc:
    ancestor.append(
        (
            token.text,
            [ancestor.text for ancestor in token.ancestors if token != ancestor],
        )
    )

ancestor = pd.DataFrame(ancestor, columns=("TOKEN_TEXT", "ANCESTORS"))
show(ancestor)
# -

# the zero index picks the first token head of multiple if the syntactic parsing was erroneous
root = [token for token in short_doc if token.head == token][0]
subject = list(root.lefts)[0]
for descendant in subject.subtree:
    print(descendant.text, [ancestor.text for ancestor in descendant.ancestors])

subject1 = list(root.lefts)
subject2 = list(root.rights)
print(subject1, "\n", subject2)

# ## Named-Entity Recognition (NER)

for ent in doc.ents:
    print("{:6} {:50}".format(ent.label_, ent.text))

displacy_visual(doc, style="ent")

# # Sentiment Analysis with TextBlob
# spaCy does not ship with sentiment lexicons, therefore I chose TextBlob to lookup sentiment values for German language. TextBlob itself is another NLP library, the TextBlobDE addition provides the German sentiment lexicon and is very easy to use. At this time, subjectivity scores are not integrated into the lexicon, only polarity scores ranging from -1 to 1.

# +
# Download the language model and nltk corpora for TextBlob the first time
if 1 == 0:
    os.system("python -m textblob.download_corpora")

from textblob_de import TextBlobDE as TextBlob

# -

# The next chunk looks through the entire document already used above and retrieves and lemmatized words used which are assigned either a positive or negative polarity. It's this easy.

for token in doc:
    if TextBlob(token.lemma_).sentiment[0] != 0:
        print("{:20} {:}".format(token.lemma_, TextBlob(token.lemma_).sentiment))

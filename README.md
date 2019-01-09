# Ineffable Wizardry of Structure

This project aims to generate more coherent ebooks-style sentences by modeling structures (i.e. parse trees), then filling in the words for each node in the parse tree with a word that has received the same tag as that node (potentially conditioned on other things too!)

`python sentences_to_tree.py corpus.txt` generates a variety of text files for model training from a file called `corpus.txt`, which must be a newline-separated file of sentences; the eventual model and its intermediate files are stored in `models/` at a subpath matching the basename of the corpus file, in this case, `models/corpus/`. `python make_models.py corpus.txt` generates the eventual models. `python markov.py corpus.txt` generates some sentences from those models.

If your input text is not a newline-separated file of sentences, `python text_to_clean_sentences.py corpus.txt` creates `corpus_sentences.txt` with all the sentences on their own lines, in a format appropriate for the input to `sentences_to_tree.py`. 

Current model: 
Only generates the parse trees from short-ish sentences, to avoid run-ons that are usually incoherent. `grep -E "^.{15,140}$" my_large_corpus.txt > corpus.txt`

Handpicked sentences from Wikipedia and Simple English Wikipedia are included in `sentences_handpicked_wikipedia.txt`, but you might get better results from your own, larger corpus.

## Quickstart

````
python sentences_to_tree.py sentences_handpicked_wikipedia.txt
python markov.py sentences_handpicked_wikipedia.txt
````


## TODO slash Jeremy's notes that probably are meaningless to you:
 - learning sentence structures
   - ALSO need to learn contextual near-equivalencies between phrases, e.g. PP(In the end) and ADV(finally)... maybe word2vec can do that?
     - train a skipgram model (where each word is represented by the one-hot encoding of the words around it), but where this is generated for all combinations of the parse tree, e.g. "I won in the end" has one for "I won" -> "In the end", one for "I won in the" -> "end", one for "i won in" -> "the end" (i.e. where "around" is informed by the structure of the parse tree)

 - add an additional parent in the markov that generates the sentence (or do we already have enough?)
 - alternatively condition on left sibling's TAG (which would get us contextual near-equivlancies between phrases)

problems:
  problem:
    intransitive verb getting direct objects
  problem:
    "to one's" as a whole PP (without something to be belonged there...)
    `that is my executive johnson writer on ruling's.`
    (../sentences_100_handpicked.txt, 3624116505669682885)
  problem:
    apostrophes showing up in the wrong kinds of places (e.g. "said year ' daniele")
  problem:
    subject-verb agreement (I thought fixed? maybe just bad backoff?)
    `his purported shipment get the indoor meters published to present much plans march 6-8 in indoor championships of bad licenses.`
    ../sentences_100_handpicked.txt,
    seed is 7072429039732098011
    solution?: treat `NNS` and `NN` as the same (like I do with VBS/VBP) because otherwise a "bad" VBX verb choice when paired with the wrong NN/NNS subject choice causes this agreement problem.

solution?:
  what if we conditioned each word being chosen on its tag, its parent AND on the number of right-siblings its parent has?


theme word choices with spacy's sense2vec
  (if I have a sentence theme ALREADY and a target theme, can calculate vector distance between the themes, apply that to the source word)

Hallmark: what do i get from using the short-sentence structure model and a tag to themed words word-filler-inner?
(So far, I'm using sentences_short/markov.pickle and the rest all from hallmark. That's not working great because there are fulltags called for that don't exist in the hallmark dataset.)

solved problems:

  problem: 
    `the tv news center 'm young.` 
    (this is just bad backoff in the 100-handpicked dataset)
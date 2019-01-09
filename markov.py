import markovify
from collections import Iterable
from random import choice,seed, randrange
from os.path import basename
from datetime import datetime
import pickle
from sys import argv, maxsize
from os.path import basename, join

randomseed = randrange(maxsize)
myseed = randomseed
# myseed = 1055650041252106232
seed(myseed)

start = datetime.now()

input_file = argv[1] if len(argv) >= 2 else "sentences_handpicked_wikipedia.txt"
input_file_slug = (basename(input_file) if "_" not in basename(input_file) else basename(input_file).split("_",1)[1]).replace(".txt", '') 
folder = join("models", input_file_slug)

GENSIM = True
GENSIM_TAGGED = True
if GENSIM:
  from gensim.models import Word2Vec
  model_filepath = join('..', 'model_sentences_5.5M_tagged_raw_words_trigrams_min_count_10_size_200_downsampling_0.001_cbow.bin'  if GENSIM_TAGGED else 'model_sentences_raw_words_trigrams_min_count_10_size_200_downsampling_0.001_cbow.bin')
  w2v_model = Word2Vec.load(model_filepath)
  cached_synonyms = {}
  def get_related(word):
    try:
      try: 
        similar_words = cached_synonyms[word]
      except KeyError:
        similar_words = w2v_model.wv.most_similar(positive=[word], negative=[], topn=10)
        cached_synonyms[word] = similar_words
    except KeyError: #word not in vocabulary
      similar_words = []
    return [word for word, _ in similar_words]
else:
  import spacy
  nlp = spacy.load('en_core_web_md')
  def get_related(word):
    word = nlp(word)[0]
    filtered_words = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    similarity = sorted(filtered_words, key=lambda w: word.similarity(w), reverse=True)
    choices = [w.lower_ for w in similarity[1:10]]
    # print("getting a word related to {}, chosen from {}".format(word, choices))
    return choices



def treeify(structure, filled_in_structure=[]):
  if not structure: 
    return []
  if len(structure) == 1:
    return structure[0] + ('' if not filled_in_structure else (' -> ' + filled_in_structure[0]))
  index_of_root = next(i for i, el in enumerate(structure) if isinstance(el, (str, bytes)))
  before = ['  ' + el for el in flatten([treeify(el, filled_in_structure[i] if filled_in_structure else []) for i,el in enumerate(structure[:index_of_root])])]
  after = ['  ' + el for el in flatten([treeify(el, filled_in_structure[index_of_root+1+i] if filled_in_structure else []) for i,el in enumerate(structure[index_of_root+1:])])]
  root = structure[index_of_root] + ('' if not filled_in_structure else (' -> ' + filled_in_structure[index_of_root]))
  return ['['] + before + [root] + after + [']']

def flatten(l):
  for el in l:
    if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
      yield from flatten(el)
    else:
      yield el

def load_model(filename):
  with open(join(folder, filename), 'rb') as f:
    return pickle.loads(f.read())

markov_model = load_model("structure_markov.pickle")
tags_parent_words_model = load_model("tags_parent_words.pickle")
tags_only_model = load_model("tags_only.pickle")
tags_parent_words_lsiblings_model = load_model("tags_parent_words_lsiblings.pickle")

print("stuff loaded after {} secs".format((datetime.now() - start).seconds ))

def walk_recursive(a, b):
  child = markov_model.walk((a.replace("_left", '').replace("_right", ''), b.replace("_left", '').replace("_right", '')))
  # print(a,b,child)
  if not child:
    return [b]
  if child[0] == "":
    return [b]
  return [walk_recursive(b, grandchild) for grandchild in child[0].split("|") if "_left" in grandchild] + [b] + [walk_recursive(b, grandchild) for grandchild in child[0].split("|") if "_right" in grandchild]

ROOT_CHOICES =  (["ROOT_VBD"] * 28 +\
                ["ROOT_VBZ"] * 14 +\
                ["ROOT_VBN"] * 14 +\
                ["ROOT_VB"] * 10 +\
                ["ROOT_NN"] * 8 +\
                ["ROOT_VBP"] * 8 +\
                ["ROOT_VBG"] * 3 +\
                ["ROOT_NNS"] * 2 +\
                ["ROOT_IN"] * 0
                )
 # ROOT_VBD frequency 28106
 # ROOT_VBZ frequency 14249
 # ROOT_VBN frequency 13773
 # ROOT_VB frequency 10027
 # ROOT_NN frequency  8205
 # ROOT_VBP frequency  7991
 # ROOT_VBG frequency  2949
 # ROOT_NNS frequency  2213
 # ROOT_PRP frequency  2076 # almost exclusively sentences that are just "mr."
 # ROOT_IN frequency  1459
START = '<BEGIN>'
def make_sentence_structure():
  root_tag = choice(ROOT_CHOICES)
  if root_tag in ["ROOT_VBP", "ROOT_VBZ"]:
    root_tag = 'ROOT_VBX'
  sentence = list(walk_recursive(START, root_tag))
  return sentence

VECTOR_EVERYTHING = False # setting this option to true sucks!

def word_for(tag, parent_word, **kwargs):
  """Find a word with the given tag whose parent word is..."""
  left_sibling = kwargs.get('left_sibling', None)
  parent_tag = kwargs.get('parent_tag', None)
  debug = False
  add_stuff = False
  candidates = None
  if debug:
    print("finding a word with tag {} whose parent is {} and left_sibling is {}".format(tag, parent_word, left_sibling))
  if left_sibling:
    candidates = tags_parent_words_lsiblings_model.get(parent_word + "|" + left_sibling + "|" + tag, None)
    if not candidates or VECTOR_EVERYTHING: 
      for related_word in get_related(parent_word + (('_' + parent_tag) if GENSIM_TAGGED and parent_tag else '' )):
        candidates =  tags_parent_words_lsiblings_model.get(parent_word + "|" + left_sibling + "|" + tag)
        if candidates:
          if debug:
            print("couldn't find one (including sibling); finding a word related to sibling {} with tag {}".format(related_word, tag))
          break
  if not candidates:
    if left_sibling:
      if debug:
        print("couldn't find a word with that sibling, backing off to just parent/tag")
    candidates = tags_parent_words_model.get(parent_word + "|" + tag, None)
  add = ""
  if (not candidates or VECTOR_EVERYTHING)  and parent_word != '<BEGIN>': 
    for related_word in get_related(parent_word + (('_' + parent_tag) if GENSIM_TAGGED and parent_tag else '' )):
      candidates = tags_parent_words_model.get(related_word + "|" + tag, None)
      if candidates:
        add = "*"
        if debug:
          print("couldn't find one; finding a word related to {} with tag {}".format(related_word, tag))
        break
  if not candidates:
    if debug:
      print("couldn't find one (or one related to {}); finding a word with tag {}".format(parent_word, tag))
    candidates = tags_only_model[tag] if tag in tags_only_model else {'*':1}
    add = "*"
  word = choice(list(flatten([[cand] * cnt for cand, cnt in candidates.items()])))
  if debug:
    print("found {}\n".format(word))
  return word + (add if add_stuff else '')


def recursively_fill_in_structure(structure, parent_word, **kwargs):
  root_tag = next((token for token in structure if isinstance(token, str)), None)
  # TODO: condition the choice of (LEFTCHILD, PARENT) and (RIGHTCHILD, PARENT) and (LEFTSIBLING, RIGHTSIBLING)
  # TODO: what if we conditioned each word being chosen on its tag, its parent AND on its parent's left sibling?
  # TODO: what if we conditioned each word being chosen on its tag, its parent AND on the number of right-siblings it has? (to fix intransitive words getting objects)
  # root_word = word_for(root_tag.replace("_left", '').replace("_right", ''), parent_word) # EXPERIMENT, uncomment
  root_word = word_for(
                root_tag.replace("_left", '').replace("_right", ''), 
                parent_word, 
                left_sibling=kwargs.get('prev_word', None),
                parent_tag=kwargs.get('parent_tag', None)
              )

  # return [root_word if isinstance(item, str) else recursively_fill_in_structure(item, root_word) for item in structure]    
  ret = []
  prev_word = None
  for item in structure:
    if isinstance(item, str): 
      ret.append(root_word)
      prev_word = None
    else:
      this_ret = recursively_fill_in_structure(item, 
                                               root_word, 
                                               prev_word=prev_word,
                                               parent_tag=root_tag.split("_")[-1].lower())
      prev_word = this_ret[0].replace("*", '') if len(item) == 1 else None
      ret.append(this_ret) # EXPERIMENT: left sibling, not sure if useful.
  return ret


TRUECASE = False
PRINT_TREE = True
if __name__ == "__main__":
  for n in range(0, 1):
    struct = make_sentence_structure()
    filled_in_structure = recursively_fill_in_structure(struct, START)

    if PRINT_TREE:
      treeified = treeify(struct, filled_in_structure)
      print(treeified if isinstance(treeified, (str, bytes)) else '\n'.join(treeified))

    sentence_tokens = list(flatten(filled_in_structure))
    if TRUECASE:
      import truecaser
      sentence = truecaser.truecase_tokens(sentence_tokens)
    else:
      sentence = sentence_tokens
    print(' '.join(sentence).replace(" ,", ',').replace(" .", '.').replace(" 's", "'s").replace(" - ", '-') + "\n" )
  if randomseed == myseed:
    print("seed is {}".format(myseed))
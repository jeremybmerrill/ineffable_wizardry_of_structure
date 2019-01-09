import spacy
import markovify
from os.path import basename, join, expanduser
from sys import argv, stderr
from os import makedirs
from datetime import datetime
# goal is to parse a corpus to dependency trees, so I can train a model
# (maybe just a markov model) for children and postcedents


class Fauxken:
  """At times, we mostly deal with tokens, but may want to create our own. This is a faux token
  (aka Fauxken) that has empty lefts and rights and has a user-definable text."""
  text = None
  lower_ = None
  lefts = []
  rights = []
  def __init__(self, n):
    self.text = n
    self.lower_ = n.lower()
BEGIN = Fauxken("<BEGIN>")

start = datetime.now()
nlp = spacy.load('en_core_web_lg') # TODO use en_core_web_lg 
print("spacy loaded after {} secs".format((datetime.now() - start).seconds ), file=stderr)

start = datetime.now()
def jbfmtag(token, position=None):
  return token.dep_ + "_" + token.tag_.replace("VBZ", "VBX").replace("VBP", "VBX") + ("_" + position if position else "")

# experiment
# def tag_grandchildren_line(token, parent=BEGIN, grandparent=BEGIN)
#   children_tagged = [jbfmtag(child, "left") for child in token.lefts] + [jbfmtag(child, "right") for child in token.rights]
#   return grandparent.text + "^" + parent.text + "^" + jbfmtag(token) + "^" + "|".join(children_tagged)

def tag_children_line(token, parent=BEGIN):
  children_tagged = [jbfmtag(child, "left") for child in token.lefts] + [jbfmtag(child, "right") for child in token.rights]
  return parent.text + "^" + jbfmtag(token) + "^" + "|".join(children_tagged)

def recursive_tag_children_line(token, parent=BEGIN):
  return [tag_children_line(token, parent)] + [item for sublist in [recursive_tag_children_line(child, Fauxken(jbfmtag(token))) for child in token.children] for item in sublist]


def pos_parent_line(token, parent=BEGIN):
  return parent.text + "|" + token.tag_.replace("VBZ", "VBX").replace("VBP", "VBX") + "^" + token.text

def recursive_pos_parent_line(token, parent=BEGIN):
  return [pos_parent_line(token, parent)] + [item for sublist in [recursive_pos_parent_line(child, token) for child in token.children] for item in sublist]

def tag_parent_lsibling_line(token, parent=BEGIN):
  lefts = list(parent.lefts)
  rights = list(parent.rights)
  if token in lefts:
    idx = lefts.index(token)
    lsibling = lefts[idx - 1] if idx > 0 else Fauxken('')
  elif token in rights:
    idx = rights.index(token)
    lsibling = rights[idx - 1] if idx > 0 else parent
  else:
    lsibling = parent
  return parent.text + "|" + lsibling.text + "|" + jbfmtag(token) + "^" + token.text


def recursive_tag_parent_lsibling_line(token, parent=BEGIN):
  return [tag_parent_lsibling_line(token, parent)] + [item for sublist in [recursive_tag_parent_lsibling_line(child, token) for child in token.children] for item in sublist]

def tag_parent_line(token, parent=BEGIN):
  return parent.text + "|" + jbfmtag(token) + "^" + token.text

def recursive_tag_parent_line(token, parent=BEGIN):
  return [tag_parent_line(token, parent)] + [item for sublist in [recursive_tag_parent_line(child, token) for child in token.children] for item in sublist]

def to_parse_tree(root):
  return [to_parse_tree(child) for child in root.lefts] + [jbfmtag(root)] + [to_parse_tree(child) for child in root.rights]

input_file = expanduser(argv[1]) if len(argv) >= 2 else "../sentences_100k.txt"
input_file_slug = (basename(input_file) if "_" not in basename(input_file) else basename(input_file).split("_",1)[1]).replace(".txt", '') 
folder = join("models", input_file_slug)
makedirs(folder, exist_ok=True)

with open(join(folder, "tag_children.txt"), 'w') as tag_children_output: 
  with open(join(folder, "tag_words.txt"), 'w') as tag_words_output: 
    with open(join(folder, "tags_parent_words.txt"), 'w') as tags_parent_words_output: 
      with open(join(folder, "tags_only.txt"), 'w') as tags_only_output: 
        with open(join(folder, "tree_sentences.txt"), 'w') as tree_sentences_output: # debug only!
          with open(join(folder, "tags_parent_words_lsiblings.txt"), 'w') as tags_parent_words_lsiblings_output:
            with open(input_file, 'r') as file:
              for line in file:
                line = line.replace("''", '').strip()
                # TODO: copy text cleaning stuff from another bit of this project
                if line == '':
                  continue
                doc = nlp(line)
                # <BEGIN>^ROOT_VBZ^meta_LS_left|punct_:_left|dobj_NN_right|punct_._right
                # ROOT_VBZ^meta_LS^
                # ROOT_VBZ^punct_:^
                # ROOT_VBZ^dobj_NN^det_DT_left|nummod_CD_left|nmod_IN_left|acl_VBN_right
                # dobj_NN^det_DT^
                root = next((token for token in doc if token.dep_ == "ROOT"), None)
                # print([item for sublist in recursive_tag_children_line(root) for item in sublist])
                tag_children = recursive_tag_children_line(root)

                # this doesn't work, but the goal was to get a whole input sentence and its whole parse tree (for easy grepping)
                tree_sentences_output.write(str(to_parse_tree(root)) + "^" + line + "\n")
                
                for tree_line in tag_children:
                  tag_children_output.write(tree_line + "\n" )

                # condition filling the words back in on something other than JUST the part of speech
                #      conditioning on previous word won't work (because we sometimes don't know the previous word)
                #      we can condition on the parent word

                # tags_only
                #   VBZ^includes
                #   LS^b
                #   :^-
                #   NN^charge

                # tag_words
                # <BEGIN>|VBZ^includes
                # includes|LS^b
                # includes|:^-
                # includes|NN^charge
                # charge|DT^a

                for tree_line in recursive_pos_parent_line(root):
                  tag_words_output.write(tree_line + "\n" )
                  split = tree_line.split("^")

                # tags_only
                # ROOT_VBZ^includes
                # meta_LS^b
                # punct_:^-
                # dobj_NN^charge

                # tags_parent_words
                # <BEGIN>|ROOT_VBZ^includes
                # includes|meta_LS^b
                # includes|punct_:^-
                # includes|dobj_NN^charge
                # charge|det_DT^a
                for tree_line in recursive_tag_parent_line(root):
                  tags_parent_words_output.write(tree_line + "\n" )
                  split = tree_line.split("^")
                  tags_only_output.write(split[0].split("|")[1] + "^" + split[1] + "\n")

                for tree_line in recursive_tag_parent_lsibling_line(root):
                  tags_parent_words_lsiblings_output.write(tree_line + "\n" )

print("intermediate text files written after {} secs".format((datetime.now() - start).seconds ), file=stderr)

import markovify
import pickle
from make_models import make_models
start = datetime.now()

make_models(folder)

print("models written after {} secs".format((datetime.now() - start).seconds ), file=stderr)

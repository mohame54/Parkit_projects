import json





class ArabCharTokenizer:
  def __init__(self,vocab):
    self._char2idx = vocab
    self._idx2char = dict(zip(vocab.values(),vocab.keys()))
  def __call__(self,text):
      text = "".join(text.split()).lower()
      text = [self._char2idx[i] for i in text]
      return text
  def decode(self,ids):
      return " ".join([self._idx2char[i] for i in ids if  i!=-1])
  @property
  def vocab(self):
    return self._char2idx

  def encode_batch(self,texts):
    ids = [self(i) for i in texts]
    max_len = max([ len(i) for i in ids])
    for j,i in enumerate(ids):
      if len(i)<max_len:
        i += [self._char2idx['']]*(max_len-len(i))
      ids[j] = i
    return ids
  def decocde_batch(self,ids):
      texts = [self.decode(i) for i in ids]
      return texts  
  @classmethod
  def from_vocab_file(cls,vocab_path):
       vocab = json.load(open(vocab_path))
       return cls(vocab)
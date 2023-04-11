import json


class Bbox(object):
    def __init__(self, xmin, ymin, xmax, ymax, conf):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.conf = conf

    @property
    def attrs(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def map(self, fn):
        attrs = list(map(fn, self.attrs))
        return Bbox(*attrs, conf=self.conf)

    def __getitem__(self, idxs):
        attrs = self.attrs
        attrs += [self.conf]
        return attrs[idxs]

    def __setitem__(self, idxs, vals):
        attrs_name = ["xmin", "ymin", "xmax", "ymax", "conf"]
        for i in idxs:
            if i > 5:
                raise IndexError('index not valid')
            attr = attrs_name[i]
            setattr(self, attr, vals[i])

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return self.width * self.height

    @classmethod
    def from_yolo(cls, item):
        xc, yc, w, h, conf = item
        xmin = max(xc - w // 2, 0)
        ymin = max(yc - h // 2, 0)
        xmax = xc + w // 2
        ymax = yc + h // 2
        return cls(xmin, ymin, xmax, ymax, conf)

    @classmethod
    def from_array_tensor(cls, item):  # bbox from prediction after nms
        items = []
        if len(item.shape) > 1:
            for it in item:
                attrs = it[:4].astype(int).tolist()
                conf = float(it[4])
                items.append(cls(*attrs, conf=conf))
            return items

        else:
            attrs = item[:4].astype(int).tolist()
            conf = float(item[4])
            return cls(*attrs, conf=conf)

    def __str__(self) -> str:
        return f"xmin:{self.xmin}, ymin:{self.ymin}, xmax:{self.xmax}, ymax:{self.ymax}, conf:{self.conf:.3f}"

    def __repr__(self) -> str:
        return f"xmin:{self.xmin}, ymin:{self.ymin}, xmax:{self.xmax}, ymax:{self.ymax}, conf:{self.conf:.3f}"

    def __add__(self, other):
        xmin = min(self.xmin, other.xmin)
        ymin = min(self.ymin, other.ymin)
        xmax = max(self.xmax, other.xmax)
        ymax = max(self.ymax, other.ymax)
        return Bbox(xmin, ymin, xmax, ymax, self.conf)

    def __sub__(self, other):
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        return Bbox(xmin, ymin, xmax, ymax, self.conf)


class ArabCharTokenizer:
    def __init__(self, vocab):
        self._char2idx = vocab
        self._idx2char = dict(zip(vocab.values(), vocab.keys()))

    def __call__(self, text):
        text = "".join(text.split()).lower()
        text = [self._char2idx[i] for i in text]
        return text

    def decode(self, ids):
        return " ".join([self._idx2char[i] for i in ids if i != -1])

    @property
    def vocab(self):
        return self._char2idx

    def encode_batch(self, texts):
        ids = [self(i) for i in texts]
        max_len = max([len(i) for i in ids])
        for j, i in enumerate(ids):
            if len(i) < max_len:
                i += [self._char2idx['']] * (max_len - len(i))
            ids[j] = i
        return ids

    def decocde_batch(self, ids):
        texts = [self.decode(i) for i in ids]
        return texts

    @classmethod
    def from_vocab_file(cls, vocab_path):
        vocab = json.load(open(vocab_path))
        return cls(vocab)

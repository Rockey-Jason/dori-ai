"""Lossless byte-level BPE tokenizer implemented from scratch.

Design goals: deterministic, reversible for arbitrary UTF-8 text, and safe on
unseen characters. Special tokens are kept as ordinary byte text in the corpus;
chat uses the literal tags as part of the prompt format.
"""
import json, re
from collections import Counter
from pathlib import Path

class BPETokenizer:
    VERSION = 3
    SPECIAL = ['<pad>','<bos>','<eos>','<unk>']
    def __init__(self, vocab_size=512):
        vocab_size=int(vocab_size)
        if vocab_size < 300: raise ValueError('vocab_size must be >= 300 for byte BPE')
        if vocab_size > 4096: raise ValueError('vocab_size must be <= 4096')
        self.target_vocab_size=vocab_size
        self.special=list(self.SPECIAL)
        self.tokens=[]; self.stoi={}; self.itos={}; self.merges=[]

    @property
    def vocab_size(self): return len(self.tokens)

    @staticmethod
    def _segments(text):
        # Keep whitespace as separate units so decoding is exactly lossless.
        return re.findall(r'\s+|[^\s]+', text, flags=re.UNICODE)

    def train(self, text):
        if not isinstance(text,str) or not text: raise ValueError('training text is empty')
        # Each non-whitespace segment is represented by raw byte tokens. This
        # guarantees that every UTF-8 character remains representable.
        freq=Counter(self._segments(text))
        seqs=[]
        for seg,n in freq.items():
            b=seg.encode('utf-8')
            seq=tuple(bytes([x]) for x in b)
            seqs.append([list(seq), int(n)])
        base=[bytes([i]) for i in range(256)]
        self.tokens=list(self.special)+base
        self.merges=[]

        # Merge over unique segments, not over the whole corpus. For the Dori
        # corpus this makes training fast while retaining deterministic BPE.
        while len(self.tokens) < self.target_vocab_size:
            counts=Counter()
            for parts,n in seqs:
                for a,b in zip(parts,parts[1:]): counts[(a,b)] += n
            if not counts: break
            pair,count=max(counts.items(), key=lambda kv:(kv[1], kv[0][0], kv[0][1]))
            if count < 2: break
            merged=pair[0]+pair[1]
            if merged in self.tokens: break
            self.tokens.append(merged)
            self.merges.append(pair)
            new=[]
            a,b=pair
            for parts,n in seqs:
                out=[]; i=0
                while i<len(parts):
                    if i+1<len(parts) and parts[i]==a and parts[i+1]==b:
                        out.append(merged); i+=2
                    else:
                        out.append(parts[i]); i+=1
                new.append([out,n])
            seqs=new
        self._rebuild(); return self

    def _rebuild(self):
        self.stoi={t:i for i,t in enumerate(self.tokens)}
        self.itos={i:t for i,t in enumerate(self.tokens)}
        self.byte_to_id={bytes([i]):4+i for i in range(256)}

    def _encode_segment(self,seg):
        parts=[bytes([x]) for x in seg.encode('utf-8')]
        for a,b in self.merges:
            merged=a+b; out=[]; i=0
            while i<len(parts):
                if i+1<len(parts) and parts[i]==a and parts[i+1]==b:
                    out.append(merged); i+=2
                else:
                    out.append(parts[i]); i+=1
            parts=out
        return [self.stoi[p] for p in parts]

    def encode(self,text,add_special=False):
        if not isinstance(text,str): raise TypeError('text must be str')
        out=[]
        for seg in self._segments(text):
            out.extend(self._encode_segment(seg))
        if add_special: out=[0]+out+[2]
        return out

    def decode(self,ids,skip_special=True):
        chunks=[]
        special_ids={0,1,2,3} if skip_special else set()
        for i in ids:
            i=int(i)
            if i in special_ids: continue
            tok=self.itos.get(i,b'')
            if isinstance(tok,bytes): chunks.append(tok)
        return b''.join(chunks).decode('utf-8',errors='replace')

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        # bytes are stored losslessly as hex.
        data={'version':self.VERSION,'target_vocab_size':self.target_vocab_size,
              'special':self.special,
              'tokens':[('b:'+t.hex()) if isinstance(t,bytes) else ('s:'+t) for t in self.tokens],
              'merges':[[a.hex(),b.hex()] for a,b in self.merges]}
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        if int(d.get('version',0)) != cls.VERSION: raise ValueError('Unsupported tokenizer version')
        x=cls(int(d['target_vocab_size'])); x.special=list(d['special'])
        x.tokens=[(bytes.fromhex(t[2:]) if t.startswith('b:') else t[2:]) for t in d['tokens']]
        x.merges=[(bytes.fromhex(a),bytes.fromhex(b)) for a,b in d['merges']]
        x._rebuild(); return x

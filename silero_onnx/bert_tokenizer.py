import os
import re
import json
import collections
import unicodedata

class BasicTokenizer:   
    def __init__(self, never_split=None, tokenize_chinese_chars=True):
        self.never_split = set(never_split) if never_split else set()
        self.tokenize_chinese_chars = tokenize_chinese_chars
    
    def tokenize(self, text, never_split=None):
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)
        
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        
        orig_tokens = text.strip().split()
        split_tokens = []
        for token in orig_tokens:
            if token in never_split:
                split_tokens.append(token)
            else:
                split_tokens.extend(self._run_split_on_punc(token, never_split))
        
        output_tokens = " ".join(split_tokens).split()
        return output_tokens

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    
    def _is_control(self, char):
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False
    
    def _is_whitespace(self, char):
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False
    
    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    
    def _is_chinese_char(self, cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or
            (cp >= 0x2B740 and cp <= 0x2B81F) or
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
        return False
    
    def _run_split_on_punc(self, token, never_split=None):
        never_split = never_split or set()
        if token in never_split:
            return [token]
        
        chars = list(token)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        
        return ["".join(x) for x in output]
    
    def _is_punctuation(self, char):
        cp = ord(char)
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        
        return False


class WordpieceTokenizer:
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
    
    def tokenize(self, text):
        output_tokens = []
        for token in text.split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                
                if cur_substr is None:
                    is_bad = True
                    break
                
                sub_tokens.append(cur_substr)
                start = end
            
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        
        return output_tokens


class SimpleBertTokenizer:
    def __init__(self, tokenizer_dir):
        tokenizer_config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
        with open(tokenizer_config_path, 'r') as f:
            tokenizer_config = json.load(f)

        # Main Tokenizer Params
        never_split = tokenizer_config.get("never_split", [])
        if never_split is None:
                never_split = []
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenizer_config.get("tokenize_chinese_chars", True)
        self.model_max_length = tokenizer_config.get("model_max_length", 2048)
        self.clean_up_tokenization_spaces = tokenizer_config.get("clean_up_tokenization_spaces", True)

        # Load Special Tokens
        special_tokens_path = os.path.join(tokenizer_dir, "special_tokens_map.json")
        with open(special_tokens_path, 'r') as f:
            special_tokens = json.load(f)
        self.unk_token = special_tokens.get("unk_token", "[UNK]")
        self.sep_token = special_tokens.get("sep_token", "[SEP]")
        self.pad_token = special_tokens.get("pad_token", "[PAD]")
        self.cls_token = special_tokens.get("cls_token", "[CLS]")
        self.mask_token = special_tokens.get("mask_token", "[MASK]")

        # Load Vocab
        vocab_path = os.path.join(tokenizer_dir, "vocab.txt")
        self.vocab = self.load_vocab(vocab_path)

        # Load Added Tokens
        added_tokens_path = os.path.join(tokenizer_dir, "added_tokens.json")
        with open(added_tokens_path, 'r', encoding='utf-8') as f:
            added_tokens = json.load(f)
        for token, token_id in added_tokens.items():
            self.vocab[token] = token_id

        self.ids_to_tokens = collections.OrderedDict(
            [(ids, token) for token, ids in self.vocab.items()]
        )

        # Set Special Tokens IDs
        self.unk_token_id = self.vocab.get(self.unk_token)
        self.sep_token_id = self.vocab.get(self.sep_token)
        self.pad_token_id = self.vocab.get(self.pad_token)
        self.cls_token_id = self.vocab.get(self.cls_token)
        self.mask_token_id = self.vocab.get(self.mask_token)
        self.homo_start_id = self.vocab.get('[HOMO]')
        self.homo_end_id = self.vocab.get('[/HOMO]')

        # Init Tokenizers
        self.basic_tokenizer = BasicTokenizer(
            never_split=self.never_split,
            tokenize_chinese_chars=self.tokenize_chinese_chars
        )
        
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=self.unk_token
        )

    def load_vocab(self, vocab_file):
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    def tokenize(self, text):
        tokens = self.basic_tokenizer.tokenize(text, never_split=self.never_split)
        tokens = self.wordpiece_tokenizer.tokenize(" ".join(tokens))
        return tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        return ids

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.ids_to_tokens.get(ids, self.unk_token)
        tokens = [self.ids_to_tokens.get(id_, self.unk_token) for id_ in ids]
        return tokens
    
    def encode(self, text):
        tokens = self.tokenize(text)
        tokens = [self.cls_token] + tokens + [self.sep_token]
        ids = self.convert_tokens_to_ids(tokens)
        return ids

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=None):
        if clean_up_tokenization_spaces is None:
            clean_up_tokenization_spaces = self.clean_up_tokenization_spaces
        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            special_tokens = {self.cls_token, self.sep_token, self.pad_token, self.mask_token, self.unk_token}
            tokens = [token for token in tokens if token not in special_tokens]
        # Remove ##
        text = ""
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                text += token[2:]
            else:
                if i != 0:
                    text += " "
                text += token
        # Clean spaces if needed
        if clean_up_tokenization_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r' \.', '.', text)
            text = re.sub(r' \?', '?', text)
            text = re.sub(r' \!', '!', text)
            text = re.sub(r' ,', ',', text)
            text = re.sub(r' ;', ';', text)
            text = re.sub(r' :', ':', text)
        return text

    def __call__(self, text):
        return self.encode(text=text)

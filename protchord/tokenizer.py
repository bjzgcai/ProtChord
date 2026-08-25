"""Tag-aware tokenizer for the public molecule/protein vocabulary."""
from __future__ import annotations
import re
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

class ProtChordTokenizer(PreTrainedTokenizerFast):
    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]
    _tags = ("<mol>", "</mol>", "<num>", "<protein>", "</protein>")
    _known = {"mol", "protein", "product", "reactants", "fragA", "fragB", "reagent", "antibody"}
    _smiles_re = re.compile(r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
    _tag_re = re.compile(r"</?(?:mol|protein|product|reactants|fragA|fragB|reagent|antibody)>")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_special_tokens = True

    def _entity(self, text: str, prefix: str, kind: str):
        tokens = self._smiles_re.findall(text) if kind == "smiles" else list(text)
        return [f"<{prefix}>{t}" for t in tokens if t]

    def _by_tag(self, span: str, tag: str, **kwargs):
        if tag in {"mol", "product", "reactants", "fragA", "fragB", "reagent"}:
            return self._entity(span.strip(), "m", "smiles")
        if tag in {"protein", "antibody"}:
            return self._entity(span.strip(), "a", "protein")
        return super().tokenize(span, **kwargs)

    def _tokenize(self, text, **kwargs):
        result, current, last = [], None, 0
        for match in self._tag_re.finditer(text):
            raw = match.group()
            tag = raw[2:-1] if raw.startswith("</") else raw[1:-1]
            if raw.startswith("</"):
                if current != tag:
                    raise ValueError(f"tag mismatch: {tag} != {current}")
                result.extend(self._by_tag(text[last:match.start()], tag, **kwargs))
                result.append(raw); current = None
            else:
                if current is not None:
                    raise ValueError(f"nested tag: {tag}")
                result.extend(super().tokenize(text[last:match.start()], **kwargs))
                result.append(raw); current = tag
            last = match.end()
        if last < len(text):
            result.extend(self._by_tag(text[last:], current, **kwargs) if current else super().tokenize(text[last:], **kwargs))
        return result

    def convert_tokens_to_string(self, tokens):
        tokens = [t.replace("<m>", "").replace("<a>", "") for t in tokens]
        return super().convert_tokens_to_string(tokens)

    def encode(self, text, **kwargs):
        ids = self.convert_tokens_to_ids(self._tokenize(text))
        if kwargs.get("add_special_tokens", False):
            return [self.bos_token_id] + ids + [self.eos_token_id]
        prepend = kwargs.get("prepend_bos", True)
        if "add_special_tokens" in kwargs and not kwargs["add_special_tokens"]:
            prepend = False
        if prepend:
            ids = [self.bos_token_id] + ids
        if kwargs.get("append_eos", False):
            ids = ids + [self.eos_token_id]
        return ids

"""Shared NER-as-SFT data plumbing for the four checkpoints.

NER is framed as instruction-style generation (not token classification): the
prompt is the sentence, the target is either a JSON object or inline
``type: value`` lines depending on ``output_format``.  The same schema/format/
parse helpers are used at train time and eval time, so the two never drift.

Output formats (set via ``output_format`` parameter):
  json    — ``{"type": ["val1", "val2"]}``  (structured, standard)
  inline  — ``type: val1\\ntype: val2``      (simpler, base-model friendly)

Available dataset keys (all load under ``datasets`` 4.x):
  conll2003     tomaarsen/conll2003          4 types, BIO
  few-nerd      DFKI-SLT/few-nerd supervised 8 coarse, IO
  few-nerd-fine DFKI-SLT/few-nerd supervised 66 fine, IO
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from datasets import load_dataset

VALID_OUTPUT_FORMATS = {"json", "inline"}
DEFAULT_OUTPUT_FORMAT = "json"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    hf_name: str
    label_column: str = "ner_tags"
    hf_config: str | None = None
    type_aliases: dict[str, str] = field(default_factory=dict)


DATASETS: dict[str, DatasetSpec] = {
    "conll2003": DatasetSpec(
        hf_name="tomaarsen/conll2003",
        type_aliases={
            "PER": "person", "ORG": "organization",
            "LOC": "location", "MISC": "miscellaneous",
        },
    ),
    "few-nerd": DatasetSpec(hf_name="DFKI-SLT/few-nerd", hf_config="supervised"),
    "few-nerd-fine": DatasetSpec(
        hf_name="DFKI-SLT/few-nerd",
        hf_config="supervised",
        label_column="fine_ner_tags",
    ),
}
DEFAULT_DATASET = "few-nerd"


def resolve_dataset(key: str) -> DatasetSpec:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; choose from {sorted(DATASETS)}")
    return DATASETS[key]


# ---------------------------------------------------------------------------
# Prompt templates & empty targets per format
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_JSON = (
    "You are a Named Entity Recognition system. Extract entities from the "
    "text and return a JSON object mapping entity types to arrays of values.\n\n"
    "Entity types: {types}\n\n"
    "Output format:\n"
    '{{"type1": ["value1", "value2"], "type2": ["value3"]}}\n\n'
    "Rules:\n"
    "- Only include entity types that have matches in the text.\n"
    "- Extract entity values exactly as they appear in the text.\n"
    "- Do not infer or guess entities that are not explicitly present.\n"
    "- If no entities are found, return an empty JSON object: {{}}\n"
    "- Output valid JSON only.\n\n"
    "Text: {text}\n\n"
    "Entities:\n"
)

PROMPT_TEMPLATE_INLINE = (
    "Extract the named entities from the text. For each entity write one line "
    '"type: entity", where type is one of: {types}. '
    'If there are no entities, write "none".\n\n'
    "Text: {text}\n\n"
    "Entities:\n"
)

EMPTY_TARGET_JSON = "{}"
EMPTY_TARGET_INLINE = "none"


def _get_format_settings(output_format: str):
    if output_format == "json":
        return PROMPT_TEMPLATE_JSON, EMPTY_TARGET_JSON
    elif output_format == "inline":
        return PROMPT_TEMPLATE_INLINE, EMPTY_TARGET_INLINE
    else:
        raise ValueError(
            f"unknown output_format {output_format!r}; "
            f"choose from {sorted(VALID_OUTPUT_FORMATS)}"
        )


# ---------------------------------------------------------------------------
# Tag schema (BIO or IO)
# ---------------------------------------------------------------------------

def _parse_tag(tag: str) -> tuple[str | None, str]:
    if tag == "O":
        return None, "O"
    if len(tag) > 2 and tag[1] == "-" and tag[0] in "BIES":
        return tag[0], tag[2:]
    return None, tag


@dataclass
class NERSchema:
    id2tag: dict[int, str]
    type_aliases: dict[str, str]
    output_format: str = DEFAULT_OUTPUT_FORMAT

    def __post_init__(self) -> None:
        if self.output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"unknown output_format {self.output_format!r}; "
                f"choose from {sorted(VALID_OUTPUT_FORMATS)}"
            )
        codes = sorted(
            {_parse_tag(t)[1] for t in self.id2tag.values() if t != "O"}
        )
        self.code_to_canon = {
            c: self.type_aliases.get(c, c.lower()) for c in codes
        }
        self.types = sorted(set(self.code_to_canon.values()))
        self._prompt_template, self._empty_target = _get_format_settings(
            self.output_format
        )

    def entities(
        self, tokens: list[str], tag_ids: list[int]
    ) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        cur_code: str | None = None
        buf: list[str] = []

        def flush() -> None:
            nonlocal cur_code, buf
            if cur_code is not None and buf:
                out.append((
                    self.code_to_canon.get(cur_code, cur_code.lower()),
                    " ".join(buf),
                ))
            cur_code, buf = None, []

        for token, tag_id in zip(tokens, tag_ids):
            prefix, code = _parse_tag(self.id2tag[int(tag_id)])
            if code == "O":
                flush()
            elif prefix in ("B", "S"):
                flush()
                cur_code, buf = code, [token]
            elif code == cur_code:
                buf.append(token)
            else:
                flush()
                cur_code, buf = code, [token]
        flush()
        return out

    def gold_set(
        self, tokens: list[str], tag_ids: list[int]
    ) -> set[tuple[str, str]]:
        return {(t, s.lower()) for t, s in self.entities(tokens, tag_ids)}

    # --- Format-aware target/prompt/parse ---

    def target_text(self, tokens: list[str], tag_ids: list[int]) -> str:
        ents = self.entities(tokens, tag_ids)
        if not ents:
            return self._empty_target
        if self.output_format == "json":
            grouped: dict[str, list[str]] = defaultdict(list)
            for t, s in ents:
                grouped[t].append(s)
            return json.dumps(grouped, ensure_ascii=False)
        else:
            return "\n".join(f"{t}: {s}" for t, s in ents)

    def build_prompt(self, tokens: list[str]) -> str:
        return self._prompt_template.format(
            types=", ".join(self.types),
            text=" ".join(tokens),
        )

    def parse(self, text: str) -> set[tuple[str, str]]:
        if self.output_format == "json":
            return self._parse_json(text)
        else:
            return self._parse_inline(text)

    def _parse_json(self, text: str) -> set[tuple[str, str]]:
        valid = set(self.types)
        found: set[tuple[str, str]] = set()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return found
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    canon = k.strip().lower()
                    if canon not in valid:
                        continue
                    if isinstance(v, list):
                        for item in v:
                            s = str(item).strip()
                            if s:
                                found.add((canon, s.lower()))
                    elif isinstance(v, str) and v.strip():
                        found.add((canon, v.strip().lower()))
        except (json.JSONDecodeError, ValueError):
            pass
        return found

    def _parse_inline(self, text: str) -> set[tuple[str, str]]:
        valid = set(self.types)
        found: set[tuple[str, str]] = set()
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.lower() == EMPTY_TARGET_INLINE or ":" not in line:
                continue
            type_part, _, surface = line.partition(":")
            canon_type, surface = type_part.strip().lower(), surface.strip()
            if canon_type in valid and surface:
                found.add((canon_type, surface.lower()))
        return found


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_split(spec: DatasetSpec, split: str, streaming: bool = False):
    if spec.hf_config:
        return load_dataset(
            spec.hf_name, spec.hf_config, split=split, streaming=streaming,
        )
    return load_dataset(spec.hf_name, split=split, streaming=streaming)


def load_schema(dataset_key: str, split: str = "train",
                output_format: str = DEFAULT_OUTPUT_FORMAT) -> NERSchema:
    spec = resolve_dataset(dataset_key)
    features = _load_split(spec, split, streaming=True).features
    names = features[spec.label_column].feature.names
    return NERSchema(
        id2tag={i: n for i, n in enumerate(names)},
        type_aliases=spec.type_aliases,
        output_format=output_format,
    )


def load_split(dataset_key: str, split: str, limit: int | None = None,
               output_format: str = DEFAULT_OUTPUT_FORMAT):
    spec = resolve_dataset(dataset_key)
    raw = _load_split(spec, split)
    if limit is not None:
        raw = raw.select(range(min(limit, len(raw))))
    names = raw.features[spec.label_column].feature.names
    schema = NERSchema(
        id2tag={i: n for i, n in enumerate(names)},
        type_aliases=spec.type_aliases,
        output_format=output_format,
    )
    return raw, schema, spec.label_column


def build_examples(
    tokenizer,
    *,
    dataset_key: str = DEFAULT_DATASET,
    split: str = "train",
    max_seq_len: int = 512,
    limit: int | None = None,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
):
    """Return a tokenized, prompt-masked ``datasets.Dataset`` plus the schema.

    ``output_format`` is ``"json"`` or ``"inline"`` — controls the prompt
    template, target serialization, and parse logic.
    """
    raw, schema, label_column = load_split(
        dataset_key, split, limit, output_format=output_format,
    )

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos = tokenizer.eos_token_id
    n_truncated = 0

    def _encode(example):
        nonlocal n_truncated
        prompt = schema.build_prompt(example["tokens"])
        target = schema.target_text(example["tokens"], example[label_column])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [eos]
        input_ids = bos + prompt_ids + target_ids
        labels = [-100] * (len(bos) + len(prompt_ids)) + target_ids
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[-max_seq_len:]
            labels = labels[-max_seq_len:]
            n_truncated += 1
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    encoded = raw.map(
        _encode, remove_columns=raw.column_names, desc=f"tokenizing {split}",
    )
    if n_truncated:
        print(
            f"[ner-data] warning: {n_truncated}/{len(encoded)} examples exceeded "
            f"max_seq_len={max_seq_len} and were left-truncated"
        )
    return encoded, schema

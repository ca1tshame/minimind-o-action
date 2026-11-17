"""Fail-closed language SFT data pipeline with explicit role and token boundaries."""
import copy
import hashlib
import json
import unicodedata
from pathlib import Path

import torch
from torch.utils.data import Dataset

RESERVED = (
    "<|im_start|>", "<|im_end|>", "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
)


class SFTSchemaError(ValueError):
    pass


class SFTTooLong(SFTSchemaError):
    pass


def _json_value(value, field):
    value = copy.deepcopy(value)
    if isinstance(value, str):
        def unique(pairs):
            out = {}
            for key, item in pairs:
                if key in out:
                    raise SFTSchemaError("duplicate JSON key in " + field)
                out[key] = item
            return out

        try:
            value = json.loads(
                value, object_pairs_hook=unique,
                parse_constant=lambda x: (_ for _ in ()).throw(SFTSchemaError("nonfinite JSON")),
            )
        except (ValueError, TypeError) as exc:
            raise SFTSchemaError("invalid JSON in " + field) from exc
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SFTSchemaError("invalid structure in " + field) from exc
    if any(marker in raw for marker in RESERVED):
        raise SFTSchemaError("reserved delimiter in " + field)
    return value


def _verified_assistant_spans(ids, tokenizer, messages, tools):
    """Return complete assistant spans after validating every rendered role."""
    roles = []
    if tools is not None or (messages and messages[0]["role"] == "system"):
        roles.append("system")
    previous = None
    for message in messages:
        role = message["role"]
        if role == "system":
            if previous is not None:
                raise ValueError("unexpected system position")
        elif role == "tool":
            if previous != "tool":
                roles.append("user")
        else:
            roles.append(role)
        previous = role
    heads = {
        role: tokenizer(tokenizer.bos_token + role + "\n", add_special_tokens=False)["input_ids"]
        for role in ("system", "user", "assistant")
    }
    end = tokenizer(tokenizer.eos_token + "\n", add_special_tokens=False)["input_ids"]
    if not end or any(not x for x in heads.values()):
        raise ValueError("invalid delimiters")
    spans, pos = [], 0
    for role in roles:
        head = heads[role]
        if ids[pos:pos + len(head)] != head:
            raise ValueError("rendered role boundary mismatch")
        start = pos + len(head)
        cursor = start
        while cursor < len(ids) and ids[cursor:cursor + len(end)] != end:
            if ids[cursor] == tokenizer.bos_token_id:
                raise ValueError("nested role boundary")
            cursor += 1
        if cursor >= len(ids):
            raise ValueError("missing role end")
        pos = cursor + len(end)
        if role == "assistant":
            spans.append((start, pos))
    if pos != len(ids):
        raise ValueError("unaccounted token suffix")
    if len(spans) != sum(m["role"] == "assistant" for m in messages):
        raise ValueError("assistant count mismatch")
    return spans


def normalize_record(record):
    if not isinstance(record, dict) or not isinstance(record.get("conversations"), list) or not record["conversations"]:
        raise SFTSchemaError("conversations must be a nonempty list")
    for key, value in record.items():
        if key not in ("conversations", "id", "source_id") and value not in (None, "", [], {}):
            raise SFTSchemaError("unsupported top-level field: " + key)
    normalized, seen_user, pending, needs_final, defined = [], False, 0, False, None
    for turn, message in enumerate(record["conversations"]):
        if not isinstance(message, dict):
            raise SFTSchemaError("message must be object")
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant", "tool"):
            raise SFTSchemaError("unsupported role: " + str(role))
        if role == "system" and turn != 0:
            raise SFTSchemaError("system message must be first")
        if role == "user":
            seen_user = True
        if role == "assistant" and not seen_user:
            raise SFTSchemaError("assistant requires an earlier user message")
        if not isinstance(content, str):
            raise SFTSchemaError("empty or non-string message content")
        tools, calls = message.get("tools"), message.get("tool_calls")
        if isinstance(tools, str) and tools:
            tools = _json_value(tools, "tools")
        if isinstance(calls, str) and calls:
            calls = _json_value(calls, "tool_calls")
        if tools not in (None, "", [], {}):
            if role != "system" or turn != 0:
                raise SFTSchemaError("tools only permitted in initial system")
            tools = _json_value(tools, "tools")
            if not isinstance(tools, list) or not tools:
                raise SFTSchemaError("tools must be a nonempty list")
            defined = set()
            for tool in tools:
                if not isinstance(tool, dict):
                    raise SFTSchemaError("invalid tool definition")
                fn = tool.get("function", tool)
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
                    raise SFTSchemaError("invalid tool name")
                if fn["name"] in defined:
                    raise SFTSchemaError("duplicate tool name")
                defined.add(fn["name"])
        else:
            tools = None
        if calls not in (None, "", [], {}):
            if role != "assistant" or pending:
                raise SFTSchemaError("unexpected tool_calls")
            calls = _json_value(calls, "tool_calls")
            if not isinstance(calls, list) or not calls or defined is None:
                raise SFTSchemaError("tool_calls require declared tools")
            for call in calls:
                fn = call.get("function", call) if isinstance(call, dict) else None
                if not isinstance(fn, dict) or fn.get("name") not in defined or "arguments" not in fn:
                    raise SFTSchemaError("invalid tool call")
                arguments = _json_value(fn["arguments"], "arguments")
                if not isinstance(arguments, dict):
                    raise SFTSchemaError("tool arguments must be object")
                fn["arguments"] = arguments
            pending, needs_final = len(calls), True
        else:
            calls = None
        if role == "tool":
            if pending < 1:
                raise SFTSchemaError("tool return without call")
            pending -= 1
        elif role == "assistant" and not calls:
            if pending:
                raise SFTSchemaError("missing tool return")
            needs_final = False
        elif role == "user" and (pending or needs_final):
            raise SFTSchemaError("incomplete tool interaction")
        if not content.strip() and not (tools is not None or calls is not None):
            raise SFTSchemaError("empty or non-string message content")
        for key, value in message.items():
            if key not in ("role", "content", "reasoning_content", "tools", "tool_calls") and value not in (None, "", [], {}):
                raise SFTSchemaError("unsupported nonempty message field: " + key)
        reasoning = message.get("reasoning_content")
        if "<think>" in content or "</think>" in content:
            if (role != "assistant" or reasoning not in (None, "") or not content.startswith("<think>\n")
                    or content.count("<think>") != 1 or content.count("</think>") != 1
                    or "\n</think>\n\n" not in content):
                raise SFTSchemaError("ambiguous embedded reasoning")
            reasoning, content = content[len("<think>\n"):].split("\n</think>\n\n", 1)
            if not content.strip() and calls is None:
                raise SFTSchemaError("empty assistant answer")
        if reasoning not in (None, ""):
            if role != "assistant" or not isinstance(reasoning, str):
                raise SFTSchemaError("reasoning must be an assistant string")
            if any(marker in reasoning for marker in RESERVED):
                raise SFTSchemaError("reserved delimiter in reasoning")
        if any(marker in content for marker in RESERVED):
            raise SFTSchemaError("reserved delimiter in content")
        item = {"role": role, "content": unicodedata.normalize("NFC", content)}
        if reasoning not in (None, ""):
            item["reasoning_content"] = unicodedata.normalize("NFC", reasoning)
        if tools is not None:
            item["tools"] = tools
        if calls is not None:
            item["tool_calls"] = calls
        normalized.append(item)
    if pending or needs_final:
        raise SFTSchemaError("incomplete tool interaction")
    if normalized[-1]["role"] != "assistant":
        raise SFTSchemaError("conversation ends without assistant answer")
    if not any(m["role"] == "assistant" for m in normalized):
        raise SFTSchemaError("no assistant message")
    return normalized


def conversation_id(messages):
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(canonical).hexdigest()


def split_for_id(sample_id, seed=42, val_percent=1):
    if not 1 <= val_percent <= 99:
        raise ValueError("val_percent must be 1..99")
    digest = hashlib.sha256((str(seed) + ":" + sample_id).encode("ascii")).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 100 < val_percent else "train"


def build_index(path, seed=42, val_percent=1):
    """Read JSONL incrementally, reject invalid rows, dedupe exact NFC conversations."""
    path = Path(path)
    offsets = {"train": [], "validation": []}
    seen, duplicates, raw_sha, rows = set(), 0, hashlib.sha256(), 0
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            raw_sha.update(line)
            rows += 1
            try:
                record = json.loads(line)
                messages = normalize_record(record)
            except Exception as exc:
                raise SFTSchemaError("row %d: %s" % (rows, exc)) from exc
            sample_id = conversation_id(messages)
            if sample_id in seen:
                duplicates += 1
                continue
            seen.add(sample_id)
            offsets[split_for_id(sample_id, seed, val_percent)].append(
                {"offset": offset, "sample_id": sample_id, "row": rows}
            )
    if not offsets["train"] or not offsets["validation"]:
        raise SFTSchemaError("split has empty train or validation")
    return {
        "data_sha256": raw_sha.hexdigest(), "rows": rows,
        "duplicates": duplicates, "seed": seed, "val_percent": val_percent,
        "splits": offsets,
    }


def encode_record(record, tokenizer, max_length=512, overflow="reject"):
    if max_length < 2 or overflow not in ("reject", "truncate_complete_assistant"):
        raise ValueError("invalid length policy")
    messages = normalize_record(record)
    tools = messages[0].get("tools") if messages and messages[0]["role"] == "system" else None
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, tools=tools
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if tokenizer.pad_token_id in ids:
        raise SFTSchemaError("pad/unknown token appears inside unpadded sample")
    try:
        spans = _verified_assistant_spans(ids, tokenizer, messages, tools)
    except ValueError as exc:
        raise SFTSchemaError("template boundary: " + str(exc)) from exc
    truncated = False
    if len(ids) > max_length:
        if overflow == "reject":
            raise SFTTooLong("token sequence exceeds max_length")
        assistants = [m for m in messages if m["role"] == "assistant"]
        cut = max(
            (end for (_, end), msg in zip(spans, assistants)
             if end <= max_length and not msg.get("tool_calls")),
            default=0,
        )
        if not cut:
            raise SFTTooLong("no complete assistant span fits max_length")
        ids = ids[:cut]
        spans = [(s, e) for s, e in spans if e <= cut]
        truncated = True
    labels = [-100] * len(ids)
    for start, end in spans:
        labels[start:end] = ids[start:end]
    if not any(x != -100 for x in labels[1:]):
        raise SFTSchemaError("no effective shifted supervision")
    if any(not isinstance(x, int) or x < 0 or x >= len(tokenizer) for x in ids):
        raise SFTSchemaError("token ID outside vocabulary")
    return {
        "input_ids": ids, "labels": labels,
        "sample_id": conversation_id(messages),
        "supervised_tokens": sum(x != -100 for x in labels[1:]),
        "truncated": truncated,
        "source_id": record.get("source_id", record.get("id")),
    }


class LanguageSFTDataset(Dataset):
    def __init__(self, path, index, tokenizer, max_length=512, overflow="reject"):
        self.path = Path(path)
        self.index = index
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overflow = overflow

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        item = self.index[i]
        with self.path.open("rb") as handle:
            handle.seek(item["offset"])
            record = json.loads(handle.readline())
        result = encode_record(record, self.tokenizer, self.max_length, self.overflow)
        if result["sample_id"] != item["sample_id"]:
            raise SFTSchemaError("indexed sample changed")
        result["row"] = item["row"]
        return result


def collate_sft(samples, pad_id=0):
    if not samples:
        raise SFTSchemaError("empty batch")
    width = max(len(x["input_ids"]) for x in samples)
    inputs, labels, mask = [], [], []
    for sample in samples:
        ids, lab = sample["input_ids"], sample["labels"]
        n = len(ids)
        if n < 2 or len(lab) != n or any(y != -100 and y != z for y, z in zip(lab, ids)) or not any(y != -100 for y in lab[1:]):
            raise SFTSchemaError("invalid labels")
        inputs.append(ids + [pad_id] * (width - n))
        labels.append(lab + [-100] * (width - n))
        mask.append([1] * n + [0] * (width - n))
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "supervised_tokens": sum(sum(y != -100 for y in x[1:]) for x in labels),
        "truncated_samples": sum(bool(x.get("truncated", False)) for x in samples),
        "sample_ids": [x["sample_id"] for x in samples],
    }

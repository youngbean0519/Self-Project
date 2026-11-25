import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sentencepiece as spm
import streamlit as st
import torch
import torch.nn as nn
from kiwipiepy import Kiwi
from kobert_tokenizer import KoBERTTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, BertModel, ElectraModel

DASHBOARD = Path(__file__).resolve().parent
sys.path.insert(0, str(DASHBOARD.parent / "src"))
import preprocessing.text as text_preprocessing

ARTIFACTS = DASHBOARD / "artifacts"
DEVICE = torch.device("cpu")
BATCH_SIZE = 1

@st.cache_data
def read_selection():
    path = ARTIFACTS / "frozen_selection.json"
    return json.loads(path.read_text(encoding="utf-8"))


def configure(selection):
    global CONFIGS, SPECS, NUM_LABELS
    CONFIGS = selection["configs"]
    NUM_LABELS = len(selection["labels"])
    SPECS = {}
    for name, spec in selection["specs"].items():
        SPECS[name] = spec


def preprocess(name, texts):
    cfg = CONFIGS[name]
    if name in ['lr','svm']:
        kwargs = {k: cfg[k] for k in ['normalize_repeat','remove_symbols']}
        series = pd.Series(texts).astype(str)
    else:
        kwargs = {'normalize_repeat': cfg['preprocessing'] != 'minimal',
                  'remove_symbols': cfg['preprocessing'] == 'repeat_symbols'}
        series = pd.Series(texts).fillna('').astype(str)
    return [text_preprocessing.preprocess_text(t, **kwargs) for t in series]


class SPTokenizer:
    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode(str(text), out_type=int)

class KiwiTokenizer:
    def __init__(self, vocab_path):
        self.kiwi = Kiwi()
        self.vocab = json.loads(
            Path(vocab_path).read_text(encoding="utf-8")
        )
        self.pad_id = 0
        self.unk_id = 1
        self.vocab_size = len(self.vocab)

    def pieces(self, text):
        return [
            f"{token.form}/{token.tag}"
            for token in self.kiwi.tokenize(str(text))
        ]

    def encode(self, text):
        return [
            self.vocab.get(piece, self.unk_id)
            for piece in self.pieces(text)
        ]

class TextCNN(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, config["embed_dim"], padding_idx=0
        )
        self.kernel_sizes = config["kernel_sizes"]
        self.mask_padding = config["cnn_mask"]

        self.convs = nn.ModuleList([
            nn.Conv1d(
                config["embed_dim"],
                config["num_filters"],
                kernel_size=k
            )
            for k in self.kernel_sizes
        ])

        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(
            config["num_filters"] * len(self.kernel_sizes),
            NUM_LABELS
        )

    def forward(self, input_ids, lengths):
        x = self.embedding(input_ids).transpose(1, 2)
        outputs = []

        for conv in self.convs:
            features = torch.relu(conv(x))

            if self.mask_padding:
                starts = torch.arange(
                    features.size(2), device=features.device
                )[None, :]

                valid = starts < lengths.to(features.device)[:, None]
                features = features.masked_fill(
                    ~valid[:, None, :],
                    torch.finfo(features.dtype).min
                )

            outputs.append(features.max(dim=2).values)

        features = torch.cat(outputs, dim=1)
        return self.classifier(self.dropout(features))

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.pooling = config["pooling"]

        self.embedding = nn.Embedding(
            vocab_size, config["embed_dim"], padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=config["embed_dim"],
            hidden_size=config["hidden_dim"],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        feature_dim = config["hidden_dim"] * 2
        if self.pooling == "meanmax":
            feature_dim *= 2

        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(feature_dim, NUM_LABELS)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, _) = self.lstm(packed)

        if self.pooling == "last":
            features = torch.cat(
                [hidden[-2], hidden[-1]], dim=1
            )

        else:
            output, _ = pad_packed_sequence(
                packed_output, batch_first=True
            )

            mask = (
                torch.arange(output.size(1), device=output.device)[None, :]
                < lengths.to(output.device)[:, None]
            )

            maximum = output.masked_fill(
                ~mask[:, :, None],
                torch.finfo(output.dtype).min
            ).max(dim=1).values

            if self.pooling == "max":
                features = maximum

            elif self.pooling == "meanmax":
                mean = (
                    output.masked_fill(~mask[:, :, None], 0).sum(dim=1)
                    / lengths.to(output.device)[:, None]
                )
                features = torch.cat([mean, maximum], dim=1)

            else:
                raise ValueError(self.pooling)

        return self.classifier(self.dropout(features))

class KoBERTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            config["model_name"]
        )
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            NUM_LABELS,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # 단일 문장이므로 token_type_ids는 기본값 0 사용
            return_dict=True,
        )

        features = self.dropout(outputs.pooler_output)
        return self.classifier(features)

class KoELECTRAClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.electra = ElectraModel.from_pretrained(
            config["model_name"]
        )
        self.dropout = nn.Dropout(config["dropout"])
        # KoBERT의 pooler에 대응하는 학습 가능한 문장 변환층
        self.dense = nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
        self.classifier = nn.Linear(
            self.electra.config.hidden_size,
            NUM_LABELS,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # 단일 문장이므로 token_type_ids는 기본값 0 사용
            return_dict=True,
        )

        features = torch.tanh(self.dense(outputs.last_hidden_state[:, 0, :]))
        features = self.dropout(features)
        return self.classifier(features)

def load_bundle(name):
    cfg = CONFIGS[name]
    folder = ARTIFACTS / name
    if name in ['lr','svm']:
        model = joblib.load(folder / SPECS[name]['file'])
        vectorizer = joblib.load(folder / 'tfidf_vectorizer.joblib')
        mlb = joblib.load(folder / 'multilabel_binarizer.joblib')
        assert np.array_equal(mlb.classes_, np.arange(NUM_LABELS)), '레이블 열 순서 불일치'
        return {'model':model, 'tokenizer':vectorizer}
    if name in ['textcnn','bilstm']:
        if cfg["tokenizer"] == "kiwi":
            tokenizer = KiwiTokenizer(folder / "vocab.json")
        elif cfg["tokenizer"] in ("unigram", "bpe"):
            tokenizer = SPTokenizer(folder / "tokenizer.model")
        else:
            raise ValueError(cfg["tokenizer"])
        model = (TextCNN if name == 'textcnn' else BiLSTM)(tokenizer.vocab_size,cfg)
    elif name == 'kobert':
        tokenizer = KoBERTTokenizer.from_pretrained(cfg['model_name'])
        assert tokenizer.encode('한국어 모델을 공유합니다.') == [2,4958,6855,2046,7088,1050,7843,54,3]
        model = KoBERTClassifier(cfg)
    else:
        tokenizer = AutoTokenizer.from_pretrained(folder / 'tokenizer', use_fast=True)
        model = KoELECTRAClassifier(cfg)
    if name in ['kobert','koelectra']:
        tokenizer.padding_side = 'right'; tokenizer.truncation_side = 'right'
    state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return {'model':model,'tokenizer':tokenizer}

@torch.inference_mode()
def predict_scores(name, bundle, texts, batch_size=BATCH_SIZE):
    # 전처리·토큰화·CPU/GPU 전송·점수의 CPU 반환까지 포함
    cfg = CONFIGS[name]
    processed = preprocess(name,texts)
    model, tok = bundle['model'], bundle['tokenizer']
    if name in ['lr','svm']:
        x = tok.transform(processed)
        scores = model.predict_proba(x) if name == 'lr' else model.decision_function(x)
        return np.asarray(scores, dtype=np.float32)
    result = []
    for start in range(0,len(processed),batch_size):
        batch = processed[start:start+batch_size]
        if name in ['textcnn','bilstm']:
            ids = [tok.encode(t)[:cfg['max_len']] or [tok.unk_id] for t in batch]
            lengths = torch.tensor([len(x) for x in ids],dtype=torch.long)
            inputs = torch.tensor([x+[0]*(cfg['max_len']-len(x)) for x in ids],dtype=torch.long,device=DEVICE)
            logits = model(inputs,lengths)
        else:
            encoded = tok(batch,padding='max_length',truncation=True,max_length=cfg['max_len'],
                          return_attention_mask=True,return_token_type_ids=False,return_tensors='pt')
            logits = model(encoded['input_ids'], encoded['attention_mask'])
        result.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.vstack(result)



@st.cache_resource
def get_bundle(name):
    return load_bundle(name)


def predict(text, name):
    selection = read_selection()
    configure(selection)
    if name == "ensemble":
        alpha = selection["alpha_kobert"]
        bert_scores = predict_scores("kobert", get_bundle("kobert"), [text])[0]
        electra_scores = predict_scores("koelectra", get_bundle("koelectra"), [text])[0]
        scores = alpha * bert_scores + (1 - alpha) * electra_scores
        policy = selection["ensemble_name"]
    else:
        scores = predict_scores(name, get_bundle(name), [text])[0]
        policy = name

    threshold = np.asarray(selection["models"][policy]["threshold"])
    labels = selection["labels"]
    if scores.shape != (len(labels),) or not np.isfinite(scores).all():
        raise ValueError("감정 점수의 개수 또는 값이 올바르지 않습니다.")
    if threshold.shape not in ((), scores.shape):
        raise ValueError("임계값 개수가 감정 점수 개수와 일치하지 않습니다.")
    score_label = "결정 점수" if name == "svm" else "모델 점수"
    return pd.DataFrame({
        "감정": labels,
        score_label: scores,
        "선택": scores >= threshold,
    }).sort_values(score_label, ascending=False, kind="stable")

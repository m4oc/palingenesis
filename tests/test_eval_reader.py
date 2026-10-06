"""The held-out eval reads the same format as training: raw text for CPT (all `mode: pretrain` sources), chat otherwise.
A CPT run with a text eval file used to get a chat reader, find no messages and silently disable validation."""

from palingenesis.config import Config
from palingenesis.data import ChatDataset, PretrainDataset
from palingenesis.train import eval_reader
from tests.test_last_turn_integration import TOK, needs_tok


def _config(sources):
    cfg = Config()
    cfg.data.sources = sources
    cfg.data.max_seq_length = 64
    return cfg


@needs_tok
def test_cpt_eval_reads_text_and_yields_samples():
    cfg = _config([{"dataset": "x.parquet", "mode": "pretrain", "text_field": "body"}])
    rows = [{"body": "Il gatto dorme sul divano. " * 20}, {"body": "La pioggia cade su Roma da tre giorni. " * 5}]
    reader = eval_reader(cfg, TOK, rows)
    assert isinstance(reader, PretrainDataset)
    samples = list(reader)
    assert samples and all((s["labels"] != -100).any() for s in samples)


@needs_tok
def test_chat_and_mixed_runs_keep_the_chat_reader():
    msgs = [{"messages": [{"role": "user", "content": "ciao"}, {"role": "assistant", "content": "ciao!"}]}]
    for sources in ([], [{"dataset": "a", "mode": "sft"}], [{"dataset": "a", "mode": "pretrain"}, {"dataset": "b"}]):
        reader = eval_reader(_config(sources), TOK, msgs)
        assert isinstance(reader, ChatDataset)
    assert list(eval_reader(_config([]), TOK, msgs))

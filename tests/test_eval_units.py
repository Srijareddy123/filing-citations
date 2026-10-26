from filingcite.evaluate import CachedLLM, is_correct, pct, summarize_answers


def test_is_correct_ignores_commas_and_dollar_signs():
    assert is_correct("It was $14,776,353,000 shares.", [["14776353000"]])
    assert not is_correct("About 15%", [["31%"]])
    assert is_correct("Boeing had 785,347,096 and 750,074,411", [["750,074,411"], ["785,347,096"]])


def test_percentiles():
    assert pct([1, 2, 3, 4, 5], 0.5) == 3
    assert pct([], 0.5) == 0.0
    assert pct(list(range(101)), 0.95) == 95


def test_cached_llm_calls_model_once(tmp_path):
    calls = []

    class M:
        name = "m"

        def complete(self, s, u):
            calls.append(1)
            return "out"

    c = CachedLLM(M(), tmp_path / "c.jsonl")
    assert c.complete("s", "u") == c.complete("s", "u") == "out"
    assert len(calls) == 1
    assert CachedLLM(M(), tmp_path / "c.jsonl").complete("s", "u") == "out" and len(calls) == 1


def test_summarize_answers_rates():
    rows = [
        {
            "id": "a",
            "type": "single_fact",
            "answerable": True,
            "refused": False,
            "correct": True,
            "n_cites": 2,
            "n_good_cites": 1,
            "n_verified": 2,
            "latency_s": 1.0,
            "model": "m",
        },
        {
            "id": "b",
            "type": "unanswerable",
            "answerable": False,
            "refused": True,
            "correct": False,
            "n_cites": 0,
            "n_good_cites": 0,
            "n_verified": 0,
            "latency_s": 2.0,
            "model": "m",
        },
    ]
    s = summarize_answers(rows)
    assert s["answer_correctness"] == 1 and s["citation_precision"] == 0.5
    assert s["refusal_accuracy_unanswerable"] == 1 and s["false_refusal_rate_answerable"] == 0

from benchmarks.reproducibility.run_llm_repro_benchmark import build_prompt, normalize_pred


def test_normalize_pred_uses_first_token_upper():
    assert normalize_pred(" b\nextra text") == "B"


def test_build_prompt_mmlu_has_answer_key():
    prompt, gold, _ = build_prompt(
        {
            "question": "2+2?",
            "choices": ["1", "2", "3", "4"],
            "answer": 3,
        },
        "mmlu",
    )
    assert "A. 1" in prompt
    assert gold == "D"


def test_build_prompt_arc_has_labels():
    prompt, gold, _ = build_prompt(
        {
            "question": "Sky color?",
            "choices": {"label": ["A", "B"], "text": ["Blue", "Green"]},
            "answerKey": "A",
        },
        "arc_easy",
    )
    assert "A. Blue" in prompt
    assert gold == "A"


def test_build_prompt_truthfulqa_mc1_has_answer_key():
    prompt, gold, _ = build_prompt(
        {
            "question": "Which statement is true?",
            "mc1_targets": {
                "choices": ["Wrong", "Correct", "Wrong 2"],
                "labels": [0, 1, 0],
            },
        },
        "truthfulqa_mc1",
    )
    assert "A. Wrong" in prompt
    assert "B. Correct" in prompt
    assert gold == "B"


def test_normalize_pred_handles_winogrande():
    assert normalize_pred("2", "winogrande") == "2"
    assert normalize_pred("B", "winogrande") == "2"


def test_normalize_pred_handles_gorilla_function_calls():
    out = "I will call: calc_binomial_probability(n=20, k=5, p=0.6)"
    assert normalize_pred(out, "gorilla_exec_simple") == "calc_binomial_probability(n=20,k=5,p=0.6)"

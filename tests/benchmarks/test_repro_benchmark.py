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

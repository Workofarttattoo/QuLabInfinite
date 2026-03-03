from bench.repro_benchmark import BenchmarkExample, extract_choice_label, format_prompt


def test_format_prompt_includes_choices_and_question():
    ex = BenchmarkExample(
        example_id="1",
        question="What is 2+2?",
        choices=["1", "2", "3", "4"],
        answer_key="D",
        subject="math",
    )
    prompt = format_prompt(ex)
    assert "Question: What is 2+2?" in prompt
    assert "A. 1" in prompt
    assert "D. 4" in prompt


def test_extract_choice_label_handles_common_output_patterns():
    assert extract_choice_label("Final answer: C") == "C"
    assert extract_choice_label("I choose option b.") == "B"
    assert extract_choice_label("no letter") is None

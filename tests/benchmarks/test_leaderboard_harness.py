from benchmarks.reproducibility.run_leaderboard_harness import (
    _discover_new_run_dir,
    _parse_tasks_csv,
    _task_to_repro_spec,
)


def test_parse_tasks_csv_trims_and_keeps_order():
    tasks = _parse_tasks_csv(" arc_challenge, mmlu ,arc_easy ")
    assert tasks == ["arc_challenge", "mmlu", "arc_easy"]


def test_task_to_repro_spec_handles_mmlu_default_subject():
    spec = _task_to_repro_spec("mmlu", "college_physics")
    assert spec is not None
    assert spec.dataset == "mmlu"
    assert spec.mmlu_subject == "college_physics"


def test_task_to_repro_spec_handles_explicit_mmlu_subject():
    spec = _task_to_repro_spec("mmlu_high_school_physics", "college_physics")
    assert spec is not None
    assert spec.dataset == "mmlu"
    assert spec.mmlu_subject == "high_school_physics"


def test_task_to_repro_spec_rejects_unsupported_task():
    assert _task_to_repro_spec("hellaswag", "college_physics") is None


def test_task_to_repro_spec_supports_winnogrande_alias():
    spec = _task_to_repro_spec("winnogrande", "college_physics")
    assert spec is not None
    assert spec.dataset == "winogrande"


def test_task_to_repro_spec_supports_truthfulqa():
    spec = _task_to_repro_spec("truthfulqa", "college_physics")
    assert spec is not None
    assert spec.dataset == "truthfulqa_mc1"


def test_task_to_repro_spec_supports_gorilla():
    spec = _task_to_repro_spec("gorilla", "college_physics")
    assert spec is not None
    assert spec.dataset == "gorilla_exec_simple"


def test_discover_new_run_dir_prefers_new_directory(tmp_path):
    output_root = tmp_path / "runs"
    output_root.mkdir()
    old = output_root / "arc_easy_20200101_000000"
    old.mkdir()
    new = output_root / "arc_easy_20200101_000100"
    new.mkdir()

    selected = _discover_new_run_dir(output_root, "arc_easy", {"arc_easy_20200101_000000"})
    assert selected == new

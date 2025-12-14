from rcmas_testbeds.testbeds.registry import build_testbed


def test_registry_builds() -> None:
    assert build_testbed("smt-co").name == "smt-co"
    assert build_testbed("qlearning").name == "qlearning"
    assert build_testbed("smt-ne").name == "smt-ne"
    assert build_testbed("hybrid").name == "hybrid"

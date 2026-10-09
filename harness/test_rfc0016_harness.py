"""Tests for rfc0016_harness.py that need neither evaluator.

The driver's verdicts are pure functions of what each implementation printed, so the cases that
matter most are tested here without building anything: bytes that differ in a way a decoder would
not see, an implementation that answers with a disposition where an error is expected, two that
agree with each other and not with the RFC. The cases file is held to the script that writes it,
so a row's expected answer cannot be edited in place.
"""
import contextlib
import io
import json
import shlex
import tempfile
from pathlib import Path
import os
import subprocess
import sys
import unittest

import rfc0016_harness as driver

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "rfc0016_cases.json")

OUTCOME = '{"handoff":{"state":"none"},"kind":"outcome","outcomeId":"approve","reasons":[],"value":{"v":"0.10"}}'
GO_RESULT = '{"outputVersion":"2","status":"evaluated","disposition":' + OUTCOME + ',"trace":[]}\n'
PY_RESULT = '{"conformanceClaim":"none","disposition":' + OUTCOME + ',"experimental":true}\n'
GO_ERROR = json.dumps(
    {"status": "error", "evaluationError": {"class": "pack-not-conformant", "phase": "preflight"}}
)
PY_ERROR = json.dumps({"error": {"class": "pack-not-conformant", "phase": "preflight"}})


class MemberBytesTests(unittest.TestCase):
    def test_the_member_is_returned_as_it_is_written(self):
        self.assertEqual(OUTCOME, driver.member_bytes(GO_RESULT, "disposition"))
        self.assertEqual(OUTCOME, driver.member_bytes(PY_RESULT, "disposition"))

    def test_whitespace_and_member_order_inside_the_member_are_kept(self):
        spaced = '{"disposition": {"kind": "outcome", "handoff": {"state": "none"}}}'
        self.assertEqual(
            '{"kind": "outcome", "handoff": {"state": "none"}}',
            driver.member_bytes(spaced, "disposition"),
        )

    def test_an_escape_is_not_turned_into_its_character(self):
        escaped = '{"disposition":{"value":{"v":"\\u00e9"}}}'
        self.assertEqual('{"value":{"v":"\\u00e9"}}', driver.member_bytes(escaped, "disposition"))

    def test_a_member_of_that_name_inside_another_member_is_not_the_member(self):
        nested = '{"trace":{"disposition":{"kind":"inner"}},"disposition":{"kind":"outer"}}'
        self.assertEqual('{"kind":"outer"}', driver.member_bytes(nested, "disposition"))
        self.assertIsNone(driver.member_bytes('{"trace":{"disposition":{}}}', "disposition"))

    def test_a_name_inside_a_string_is_not_a_member(self):
        self.assertIsNone(driver.member_bytes('{"note":"\\"disposition\\":{}"}', "disposition"))

    def test_text_that_is_not_one_object_has_no_member(self):
        for text in ("", "not json", "[]", "null", '"disposition"', '{"disposition"', '{"disposition":'):
            with self.subTest(text=text):
                self.assertIsNone(driver.member_bytes(text, "disposition"))


class AnswerOfTests(unittest.TestCase):
    def test_go_writes_a_result_and_an_error_to_stdout(self):
        self.assertEqual(("disposition", OUTCOME), driver.answer_of(GO_RESULT, GO_RESULT, "evaluationError"))
        self.assertEqual(("error", "pack-not-conformant"), driver.answer_of(GO_ERROR, GO_ERROR, "evaluationError"))

    def test_python_writes_a_result_to_stdout_and_an_error_to_stderr(self):
        self.assertEqual(("disposition", OUTCOME), driver.answer_of(PY_RESULT, "", "error"))
        self.assertEqual(("error", "pack-not-conformant"), driver.answer_of("", PY_ERROR, "error"))

    def test_an_error_together_with_a_disposition_is_reported_as_both(self):
        both = '{"disposition":' + OUTCOME + ',"evaluationError":{"class":"malformed-input"}}'
        self.assertEqual(("both", "malformed-input"), driver.answer_of(both, both, "evaluationError"))

    def test_output_that_is_neither_is_unparsed(self):
        for text in ("", "not json", "[]", json.dumps({"error": "a string"}), json.dumps({"error": {"class": 7}})):
            with self.subTest(text=text):
                self.assertEqual("unparsed", driver.answer_of(text, text, "error")[0])


class VerdictTests(unittest.TestCase):
    def test_both_agree_with_the_rfc(self):
        answer = ("disposition", OUTCOME)
        self.assertEqual("matches-rfc", driver.verdict_of(answer, answer, answer))

    def test_one_byte_of_difference_is_a_divergence(self):
        other = ("disposition", OUTCOME.replace('"0.10"', '"0.1"'))
        self.assertEqual("DIVERGENT", driver.verdict_of(("disposition", OUTCOME), other, ("disposition", OUTCOME)))

    def test_the_same_value_written_in_another_order_is_a_divergence(self):
        reordered = '{"kind":"outcome","handoff":{"state":"none"},"outcomeId":"approve","reasons":[],"value":{"v":"0.10"}}'
        self.assertEqual(json.loads(OUTCOME), json.loads(reordered))
        self.assertEqual(
            "DIVERGENT",
            driver.verdict_of(("disposition", OUTCOME), ("disposition", reordered), ("disposition", OUTCOME)),
        )

    def test_a_disposition_against_an_error_is_a_divergence(self):
        self.assertEqual(
            "DIVERGENT",
            driver.verdict_of(("error", "malformed-input"), ("disposition", OUTCOME), ("disposition", OUTCOME)),
        )

    def test_two_classes_that_differ_are_a_divergence(self):
        self.assertEqual(
            "DIVERGENT",
            driver.verdict_of(("error", "malformed-input"), ("error", "pack-not-conformant"), ("error", "malformed-input")),
        )

    def test_both_agree_with_each_other_and_not_with_the_rfc(self):
        answer = ("error", "pack-not-conformant")
        self.assertEqual("AGREE-OFF-RFC", driver.verdict_of(answer, answer, ("error", "unsupported-required-extension")))

    def test_two_that_agree_on_a_forbidden_or_unreadable_answer_do_not_agree(self):
        for kind in ("both", "unparsed"):
            with self.subTest(kind=kind):
                answer = (kind, "malformed-input")
                self.assertEqual("DIVERGENT", driver.verdict_of(answer, answer, answer))


class CasesFileTests(unittest.TestCase):
    def setUp(self):
        with open(CASES, encoding="utf-8") as handle:
            self.text = handle.read()
        self.document = json.loads(self.text)

    def test_the_file_is_what_the_script_writes(self):
        written = subprocess.run(
            [sys.executable, os.path.join(HERE, "make_rfc0016_cases.py")],
            capture_output=True, text=True, check=True,
        ).stdout
        self.assertEqual(written, self.text)

    def test_every_row_has_one_expected_answer_and_an_origin_in_the_rfc(self):
        self.assertEqual(66, len(self.document["cases"]))
        ids = [case["id"] for case in self.document["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        for case in self.document["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(1, len(case["expected"]))
                self.assertIn(list(case["expected"])[0], ("disposition", "errorClass"))
                self.assertTrue(case["origin"].startswith(("Conformance, ", "Examples, ")))
                self.assertIsInstance(case["optIn"], bool)

    def test_an_expected_disposition_is_written_in_its_canonical_form(self):
        for case in self.document["cases"]:
            expected = case["expected"].get("disposition")
            if expected is None:
                continue
            with self.subTest(case=case["id"]):
                decoded = json.loads(expected)
                self.assertEqual(
                    expected,
                    json.dumps(decoded, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                )
                if decoded["kind"] != "outcome":
                    self.assertNotIn("value", decoded)

    def test_the_rows_that_hold_an_escape_hold_it_as_text(self):
        rows = {case["id"]: case for case in self.document["cases"]}
        backslash = chr(92)
        self.assertIn(backslash + "ud83d" + backslash + "ude00", rows["document-constant-surrogate-pair"]["pack"])
        self.assertIn(backslash + "ud800", rows["document-string-constant-unpaired-surrogate"]["pack"])
        self.assertIn(backslash + "udc00", rows["adversarial-string-fact-unpaired-surrogate"]["facts"])
        self.assertIn("amount" + backslash + "n", rows["document-name-ends-in-line-feed"]["pack"])

    def test_the_rows_of_the_examples(self):
        rows = {case["id"]: case for case in self.document["cases"]}
        examples = [case for case in self.document["cases"] if case["origin"].startswith("Examples, ")]
        self.assertEqual(11, len(examples))
        # The three outcomes of the tiers, each by a row of the one pack.
        tiers = [rows["example-tiers-" + which] for which in ("high", "standard", "decline")]
        self.assertEqual(1, len({row["pack"] for row in tiers}))
        self.assertEqual(
            ["limit-high", "limit-standard", "decline"],
            [json.loads(row["expected"]["disposition"])["outcomeId"] for row in tiers],
        )
        # The pack without the declaration is a pack of Core's: it does not hold the
        # extension's name, it has no metadata, and it is run with neither opt-in. But for
        # those it is the pack of the pass-through rows.
        declared = json.loads(rows["example-pass-through"]["pack"])
        del declared["outcomes"][0]["extensions"]
        del declared["metadata"]
        for name in ("example-pass-through-undeclared-amount-absent", "example-pass-through-undeclared-amount-a-number"):
            with self.subTest(row=name):
                row = rows[name]
                self.assertFalse(row["optIn"])
                self.assertNotIn("org.judgmentpack.", row["pack"])
                undeclared = json.loads(row["pack"])
                self.assertNotIn("metadata", undeclared)
                self.assertNotIn("value", json.loads(row["expected"]["disposition"]))
                self.assertEqual(declared, undeclared)
                self.assertEqual(rows[name.replace("-undeclared", "")]["facts"], row["facts"])
        # The RFC gives no pack for the calculated quantity, so each row of it says what
        # was chosen in writing one.
        calculated = [case for case in examples if case["origin"].startswith("Examples, a calculated quantity: ")]
        self.assertEqual(3, len(calculated))
        self.assertEqual(1, len({row["pack"] for row in calculated}))
        for row in calculated:
            with self.subTest(row=row["id"]):
                self.assertTrue(row["note"].startswith("The RFC gives no pack for this example. Chosen in writing one: "))
                self.assertIn('"fromFact": "/refund/amount"', row["pack"])
                self.assertIn('"value": "500"', row["pack"])
        # A value is copied as it was found: 500.00 is not written as 500.
        self.assertEqual(
            {"refundAmount": "500.00"},
            json.loads(rows["example-calculated-at-the-bound"]["expected"]["disposition"])["value"],
        )

    def test_every_known_row_is_a_row_and_gives_its_reason(self):
        ids = {case["id"] for case in self.document["cases"]}
        self.assertEqual(4, len(self.document["known"]))
        for name, known in self.document["known"].items():
            with self.subTest(row=name):
                self.assertIn(name, ids)
                self.assertIn(known["verdict"], ("AGREE-OFF-RFC", "DIVERGENT"))
                self.assertTrue(known["reason"])



class DriverExitBehaviorTests(unittest.TestCase):
    """Exercise main() with stand-in implementations, not just verdict helpers."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pyrepo = self.root / "python"
        package = self.pyrepo / "jps_evaluator"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        self.python_main = package / "__main__.py"
        self.go = self.root / "standin.sh"
        self.go_output = self.root / "go.json"
        self.cases_path = self.root / "cases.json"

    def exercise(self, *, go=GO_RESULT, python=PY_RESULT, known=None, rows=None):
        # The fake executable refers to an absolute output path because the driver
        # launches each case with a new, otherwise empty temporary cwd.
        self.go_output.write_text(go, encoding="utf-8")
        self.go.write_text(
            "#!/bin/sh\ncat " + shlex.quote(str(self.go_output)) + "\n",
            encoding="utf-8",
        )
        self.go.chmod(0o755)
        self.python_main.write_text(
            "import sys\nsys.stdout.write(" + repr(python) + ")\n",
            encoding="utf-8",
        )
        cases = [{
            "id": "row-1",
            "origin": "Conformance, synthetic test",
            "optIn": True,
            "pack": "{}",
            "facts": "{}",
            "expected": {"disposition": OUTCOME},
        }] if rows is None else rows
        self.cases_path.write_text(
            json.dumps({"cases": cases, "known": known or {}}),
            encoding="utf-8",
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = driver.main(
                ["rfc0016_harness.py", str(self.go), str(self.pyrepo), str(self.cases_path)]
            )
        return code, stdout.getvalue(), stderr.getvalue()

    def test_expected_verdict_exits_zero(self):
        code, stdout, stderr = self.exercise()
        self.assertEqual(code, 0, (stdout, stderr))
        self.assertIn("matches-rfc", stdout)
        self.assertNotIn("FAILED:", stdout)

    def test_unlisted_verdicts_fail_the_run(self):
        different_go = GO_RESULT.replace('"0.10"', '"0.1"')
        different_python = PY_RESULT.replace('"0.10"', '"0.1"')
        for title, go, python, verdict in (
            ("disagree", different_go, PY_RESULT, "DIVERGENT"),
            ("agree_off_rfc", different_go, different_python, "AGREE-OFF-RFC"),
        ):
            with self.subTest(kind=title):
                code, stdout, _ = self.exercise(go=go, python=python)
                self.assertEqual(code, 1)
                self.assertIn(verdict, stdout)
                self.assertIn("FAILED:", stdout)
                self.assertIn("row-1", stdout)

    def test_declared_known_divergence_can_pass(self):
        code, stdout, _ = self.exercise(
            go=GO_RESULT.replace('"0.10"', '"0.1"'),
            known={"row-1": {"verdict": "DIVERGENT", "reason": "synthetic mismatch"}},
        )
        self.assertEqual(code, 0)
        self.assertIn("known: synthetic mismatch", stdout)
        self.assertNotIn("FAILED:", stdout)

    def test_stale_known_verdict_fails(self):
        code, stdout, _ = self.exercise(
            known={"row-1": {"verdict": "DIVERGENT", "reason": "stale exception"}}
        )
        self.assertEqual(code, 1)
        self.assertIn("matches-rfc", stdout)
        self.assertIn("FAILED:", stdout)
        self.assertIn("row-1", stdout)

    def test_empty_cases_are_refused_before_running(self):
        code, stdout, stderr = self.exercise(rows=[])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("refused: the cases file holds zero rows", stderr)

    def test_a_known_name_without_a_row_is_refused(self):
        code, stdout, stderr = self.exercise(
            known={"not-a-row": {"verdict": "DIVERGENT", "reason": "typo"}}
        )
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn('refused: "known" names rows the cases file does not hold', stderr)
        self.assertIn("not-a-row", stderr)


if __name__ == "__main__":
    unittest.main()

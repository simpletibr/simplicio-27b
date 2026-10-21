"""Contract of decide/think (issue #75, F2b: gate GT "pensar sob demanda").

Pins the public behavior the notebooks and the other squads rely on: the deterministic sample, the
prompt and the answer reader, the resume rules, the scores, the paired bootstrap and the GT verdict.
Flows that touch disk, a fake HTTP endpoint or a tiny torch model are in test_flow_75.py.

Everything here runs on CPU with the standard library only.  The rows are synthetic; no benchmark
text is ever written to this repository.
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from decide.think import answers as A  # noqa: E402
from decide.think import common as C  # noqa: E402
from decide.think import compare as K  # noqa: E402
from decide.think import hosted as H  # noqa: E402
from decide.think import prompt as P  # noqa: E402
from decide.think import sample as S  # noqa: E402
from decide.think import serve as SV  # noqa: E402
from decide.think import thinker as T  # noqa: E402
from decide.think import trace as TR  # noqa: E402
import think_nb_support as NB  # noqa: E402


def opts(keys, texts=None):
    """Options of a choice question built the way the prompt builder does."""
    texts = texts or {k: f"text of {k}" for k in keys}
    return P.option_list({k: texts[k] for k in keys})


class ConstantsContract(unittest.TestCase):
    def test_pre_registered_defaults(self) -> None:
        self.assertEqual(C.SEED, 20261021)
        self.assertEqual(C.PER_BENCHMARK, 40)
        self.assertEqual(C.GT_MIN_DELTA_PTS, 10.0)
        self.assertEqual(C.BOOTSTRAP_B, 10_000)
        self.assertEqual(C.CI_LEVEL, 0.95)

    def test_knowledge_benchmarks_of_the_brief(self) -> None:
        self.assertEqual(C.DEFAULT_BENCHMARKS, (25, 28, 30, 31, 32, 33, 43, 44, 45, 57, 58))
        names = {i: C.benchmark_name(i) for i in C.DEFAULT_BENCHMARKS}
        self.assertEqual(names[25], "GPQA Diamond")
        self.assertEqual(names[45], "HLE")
        self.assertEqual(names[58], "BBH")

    def test_kit_chance_values(self) -> None:
        # decision_index/data/index-0.3.json "chance" @25c6ec7
        self.assertEqual({i: C.chance_of(i) for i in (25, 30, 31, 33, 44)},
                         {25: 0.25, 30: 0.175, 31: 0.0819, 33: 0.0131, 44: 0.5})

    def test_skill_is_the_kit_formula(self) -> None:
        self.assertAlmostEqual(C.skill(0.52, 0.25), 0.36)          # GPQA 52.0% accuracy = 36.0 skill
        self.assertEqual(C.skill(0.2, 0.25), 0.0)                   # below chance is clipped
        self.assertEqual(C.skill(1.0, 0.25), 1.0)
        self.assertEqual(C.skill(0.5, 1.0), 0.5)                    # degenerate chance does not divide by zero

    def test_decision_models(self) -> None:
        self.assertEqual(len(C.DECISION_MODELS), 8)
        self.assertEqual(len(set(C.DECISION_MODELS)), 8)


class PromptContract(unittest.TestCase):
    def test_small_choice_sets_get_positional_letters_and_keep_the_key(self) -> None:
        o = P.option_list({"(A)": "one", "(B)": "two", "(C)": "three"})
        self.assertEqual([(x.label, x.key, x.text) for x in o], [("A", "(A)", "one"), ("B", "(B)", "two"), ("C", "(C)", "three")])
        o = P.option_list({"option_0": "17", "option_1": "23"})
        self.assertEqual([(x.label, x.key) for x in o], [("A", "option_0"), ("B", "option_1")])

    def test_yes_no_and_true_false_are_their_own_labels(self) -> None:
        for crit in ({"no": "No", "yes": "Yes"}, {"Yes": "Yes", "No": "No"}, {"True": "True", "False": "False"}):
            o = P.option_list(crit)
            self.assertEqual([x.label for x in o], list(crit), crit)
            self.assertEqual([x.key for x in o], list(crit), crit)

    def test_more_than_26_options_use_the_keys_as_labels(self) -> None:
        crit = {f"a1b{i % 8 + 1}": {"uci": f"a1b{i % 8 + 1}", "san": f"M{i}"} for i in range(8)}
        crit.update({f"c{i}d{j}": {"uci": f"c{i}d{j}", "san": f"N{i}{j}"} for i in range(1, 6) for j in range(1, 6)})
        self.assertGreater(len(crit), 26)
        o = P.option_list(crit)
        self.assertEqual([x.label for x in o], [x.key for x in o])
        self.assertEqual(o[0].text, "M0")           # a chess move shows its SAN, the label stays the UCI

    def test_exactly_26_options_still_fit_in_letters(self) -> None:
        o = P.option_list({f"k{i}": f"t{i}" for i in range(26)})
        self.assertEqual(o[0].label + o[-1].label, "AZ")

    def test_state_rendering(self) -> None:
        self.assertEqual(P.render_state(""), "")
        self.assertEqual(P.render_state(None), "")
        self.assertEqual(P.render_state({}), "")
        self.assertEqual(P.render_state("  plain text  "), "plain text")
        self.assertEqual(P.render_state({"question": "2+2?"}), "question: 2+2?")
        self.assertEqual(P.render_state({"board": "a b\nc d", "n": 3}), "board:\na b\nc d\nn: 3")
        self.assertEqual(P.render_state({"xs": [1, 2]}), "xs: [1, 2]")

    def request(self, **kw):
        base = {"state": {"question": "Which city is the capital of France?"},
                "questions": {"q1": {"type": "choice", "instructions": "Pick the capital. Do not provide reasoning.",
                                     "criteria": {"A": "Paris", "B": "Rome", "C": "Berlin", "D": "Madrid"}}}}
        base.update(kw)
        return base

    def test_prompt_shows_state_instructions_and_labeled_options_verbatim(self) -> None:
        pr = P.build_prompt(self.request())
        for needle in ("question: Which city is the capital of France?", "Pick the capital. Do not provide reasoning.",
                       "[Question q1]", "A. Paris", "D. Madrid", "ANSWER q1: A"):
            self.assertIn(needle, pr.text)
        self.assertEqual(list(pr.options), ["q1"])
        self.assertEqual([o.key for o in pr.options["q1"]], ["A", "B", "C", "D"])

    def test_prompt_does_not_leak_gold_or_decisions(self) -> None:
        req = self.request(expected={"q1": "A"}, gold={"q1": ["A"]}, decisions={"d3": {"q1": {"choice": "A"}}})
        text = P.build_prompt(req).text
        for banned in ("expected", "gold", "decisions", "d3"):
            self.assertNotIn(banned, text)

    def test_empty_state_has_no_context_section(self) -> None:
        text = P.build_prompt(self.request(state={})).text
        self.assertNotIn("[Context]", text)
        self.assertIn("[Context]", P.build_prompt(self.request()).text)

    def test_reasoning_ban_in_the_question_is_reconciled_by_the_reply_format(self) -> None:
        text = P.build_prompt(self.request()).text
        self.assertIn("thinking block", text)

    def test_multi_question_request_lists_every_question_and_one_answer_line_each(self) -> None:
        req = {"state": {"paragraph": "text"}, "questions": {
            f"option_{i}": {"type": "choice", "instructions": f"Is candidate {i} right?", "criteria": {"no": "No", "yes": "Yes"}}
            for i in range(3)}}
        pr = P.build_prompt(req)
        for i in range(3):
            self.assertIn(f"[Question option_{i}]", pr.text)
            self.assertIn(f"Is candidate {i} right?", pr.text)
        self.assertIn("ANSWER option_0: no", pr.text)
        self.assertEqual(list(pr.options), ["option_0", "option_1", "option_2"])

    def test_non_choice_questions_are_refused(self) -> None:
        bad = {"state": {}, "questions": {"q": {"type": "noul", "instructions": "x"}}}
        with self.assertRaises(P.UnsupportedQuestion):
            P.build_prompt(bad)
        with self.assertRaises(ValueError):
            P.build_prompt({"state": {}, "questions": {}})

    def test_prompt_is_deterministic(self) -> None:
        self.assertEqual(P.build_prompt(self.request()).text, P.build_prompt(self.request()).text)

    def test_multiline_option_text_keeps_the_label_on_its_own_line(self) -> None:
        req = self.request()
        req["questions"]["q1"]["criteria"]["B"] = "line one\nline two\n"
        text = P.build_prompt(req).text
        self.assertIn("B. line one\n   line two\nC. Berlin", text)


class SplitReasoningContract(unittest.TestCase):
    def test_closed_block(self) -> None:
        self.assertEqual(A.split_reasoning("<think>abc</think>\n\nANSWER q1: B"), ("abc", "ANSWER q1: B"))

    def test_template_opened_the_block_so_only_the_closing_tag_is_in_the_text(self) -> None:
        self.assertEqual(A.split_reasoning("step 1\nstep 2\n</think>\n\nANSWER q1: B"), ("step 1\nstep 2", "ANSWER q1: B"))

    def test_last_closing_tag_wins(self) -> None:
        self.assertEqual(A.split_reasoning("a</think>b</think>c"), ("a</think>b", "c"))

    def test_unfinished_block_has_no_final_answer(self) -> None:
        self.assertEqual(A.split_reasoning("<think>still thinking ANSWER q1: B"), ("still thinking ANSWER q1: B", None))

    def test_no_tags_means_the_endpoint_already_separated_the_reasoning(self) -> None:
        self.assertEqual(A.split_reasoning("ANSWER q1: B"), ("", "ANSWER q1: B"))
        self.assertEqual(A.split_reasoning(""), ("", ""))
        self.assertEqual(A.split_reasoning(None), ("", None))


class ExtractContract(unittest.TestCase):
    ABCD = {"q1": opts("ABCD", {"A": "Paris", "B": "Rome", "C": "Berlin", "D": "Madrid"})}

    def one(self, text, options=None, qid="q1"):
        return A.extract(text, options or self.ABCD)[qid]

    def test_the_instructed_format(self) -> None:
        for text in ("ANSWER q1: B", "answer q1: b)", "**ANSWER q1: B**", "ANSWER q1: **B**", "ANSWER q1: (B)",
                     "ANSWER q1 = B", "ANSWER [q1]: B", "Some words.\n\nANSWER q1: B\n", "ANSWER q1: B) Rome",
                     "ANSWER q1: Option B", "ANSWER q1:\tB", "FINAL ANSWER q1: B", "ANSWER q1: \\boxed{B}"):
            self.assertEqual(self.one(text), "B", text)

    def test_last_valid_marker_wins(self) -> None:
        self.assertEqual(self.one("ANSWER q1: A\nOn reflection that is wrong.\nANSWER q1: D"), "D")
        self.assertEqual(self.one("ANSWER q1: A\nANSWER q1: B\nANSWER q1: C"), "C")

    def test_the_last_marker_decides_even_when_it_is_not_a_valid_option(self) -> None:
        # A model that changes its mind to something unreadable has not answered: reading the earlier line would be a guess.
        self.assertIsNone(self.one("ANSWER q1: B\nANSWER q1: ???"))
        self.assertIsNone(self.one("ANSWER q1: B\nANSWER q1: Z"))                 # Z is not a label of a 4-option question
        self.assertIsNone(self.one("ANSWER q1: B\nANSWER q1: E"))

    def test_a_marker_that_is_present_but_invalid_does_not_fall_back_to_a_bare_label(self) -> None:
        yn = {"q": P.option_list({"no": "No", "yes": "Yes"})}
        self.assertIsNone(A.extract("There is no causal link between them.\nANSWER q: maybe", yn)["q"])
        self.assertIsNone(self.one("Options A and B are out.\nANSWER q1: none of them"))
        self.assertEqual(A.extract("There is no causal link between them.", yn)["q"], "no")       # without a marker the last bare label is still the last resort

    def test_a_marker_with_the_value_on_the_next_line(self) -> None:
        self.assertEqual(self.one("ANSWER q1:\n**B**"), "B")
        self.assertEqual(self.one("ANSWER q1:\n\n  (C)\n"), "C")
        self.assertIsNone(self.one("ANSWER q1:\n"))

    def test_a_question_line_with_the_value_on_the_next_line(self) -> None:
        o = {f"option_{i}": P.option_list({"no": "No", "yes": "Yes"}) for i in range(2)}
        self.assertEqual(A.extract("ANSWER option_0: yes\n- option_1:\nno", o), {"option_0": "yes", "option_1": "no"})
        self.assertEqual(A.extract("ANSWER option_0: yes\n- option_1:\n\n  **no**", o)["option_1"], "no")
        self.assertIsNone(A.extract("ANSWER option_0: yes\n- option_1:", o)["option_1"])

    def test_how_each_answer_was_found(self) -> None:
        d = A.extract_detailed
        self.assertEqual(d("ANSWER q1: B", self.ABCD), {"q1": ("B", "marker")})
        self.assertEqual(d("The answer is C.", self.ABCD), {"q1": ("C", "generic")})
        self.assertEqual(d("so \\boxed{D}", self.ABCD), {"q1": ("D", "boxed")})
        self.assertEqual(d("B", self.ABCD), {"q1": ("B", "bare")})
        self.assertEqual(d("I cannot decide.", self.ABCD), {"q1": (None, None)})
        self.assertEqual(d("ANSWER q1: Z", self.ABCD), {"q1": (None, "marker")})
        self.assertEqual(d("", self.ABCD), {"q1": (None, None)})
        o = {f"option_{i}": P.option_list({"no": "No", "yes": "Yes"}) for i in range(2)}
        self.assertEqual(d("ANSWER option_0: yes\n- option_1: no", o), {"option_0": ("yes", "marker"), "option_1": ("no", "line")})

    def test_without_the_question_id_a_single_question_still_parses(self) -> None:
        for text in ("ANSWER: C", "Answer: C", "The answer is C.", "The correct answer is **C**", "Final answer - (C)",
                     "Final Answer: \\boxed{C}", "answer is C, not A", "\\boxed{C}"):
            self.assertEqual(self.one(text), "C", text)

    def test_a_bare_label_is_the_last_resort_and_the_last_one_counts(self) -> None:
        self.assertEqual(self.one("B"), "B")
        self.assertEqual(self.one("**C**"), "C")
        self.assertEqual(self.one("Options A, B and C do not work, so D"), "D")
        self.assertEqual(self.one("(A)"), "A")

    def test_lowercase_letter_needs_an_enumerator_delimiter(self) -> None:
        self.assertIsNone(self.one("ANSWER q1: a car is needed"))                # an article, not option A
        self.assertEqual(self.one("ANSWER q1: a"), "A")
        self.assertEqual(self.one("ANSWER q1: a) Paris"), "A")
        self.assertIsNone(self.one("ANSWER q1: Boston"))                        # B is not the first letter of a word

    def test_option_text_and_key_are_accepted_when_unambiguous(self) -> None:
        self.assertEqual(self.one("ANSWER q1: Berlin"), "C")
        self.assertEqual(self.one("ANSWER q1: berlin"), "C")
        o = {"answer": P.option_list({"option_0": "17", "option_1": "23", "option_2": "18"})}
        self.assertEqual(A.extract("ANSWER answer: 23", o)["answer"], "option_1")
        self.assertEqual(A.extract("ANSWER answer: option_2", o)["answer"], "option_2")
        dup = {"q1": P.option_list({"x": "same", "y": "same"})}
        self.assertIsNone(A.extract("ANSWER q1: same", dup)["q1"])

    def test_unparseable_is_none(self) -> None:
        for text in ("", "   ", "I cannot decide.", "ANSWER q1:", "ANSWER q1: ?", "ANSWER q1: Z", "ANSWER q1: both", None):
            self.assertIsNone(self.one(text), text)

    def test_unfinished_thinking_gives_nothing_even_if_a_label_is_inside(self) -> None:
        self.assertEqual(A.extract_from_text("<think>maybe ANSWER q1: B", self.ABCD), {"q1": None})
        self.assertEqual(A.extract_from_text("<think>x</think>ANSWER q1: B", self.ABCD), {"q1": "B"})
        self.assertEqual(A.extract_from_text("x\n</think>\nANSWER q1: B", self.ABCD), {"q1": "B"})

    def test_reasoning_that_mentions_labels_does_not_count(self) -> None:
        text = "<think>Maybe ANSWER q1: A? No.</think>\nANSWER q1: C"
        self.assertEqual(A.extract_from_text(text, self.ABCD), {"q1": "C"})

    def test_yes_no_labels_are_case_insensitive_and_return_the_key(self) -> None:
        o = {"q": P.option_list({"no": "No", "yes": "Yes"})}
        for text, want in (("ANSWER q: yes", "yes"), ("ANSWER q: YES.", "yes"), ("ANSWER q: No", "no"), ("ANSWER q: **no**", "no"),
                           ("ANSWER q: yes, because", "yes")):
            self.assertEqual(A.extract(text, o)["q"], want, text)
        o2 = {"q": P.option_list({"Yes": "Yes", "No": "No"})}
        self.assertEqual(A.extract("ANSWER q: yes", o2)["q"], "Yes")
        self.assertIsNone(A.extract("ANSWER q: maybe", o2)["q"])

    def test_many_questions_do_not_leak_into_each_other(self) -> None:
        o = {f"option_{i}": P.option_list({"no": "No", "yes": "Yes"}) for i in (1, 10, 2)}
        text = "ANSWER option_1: yes\nANSWER option_10: no\nANSWER option_2: yes"
        self.assertEqual(A.extract(text, o), {"option_1": "yes", "option_10": "no", "option_2": "yes"})
        only10 = "ANSWER option_10: no"
        self.assertEqual(A.extract(only10, o), {"option_1": None, "option_10": "no", "option_2": None})

    def test_relaxed_lines_for_many_questions(self) -> None:
        o = {f"option_{i}": P.option_list({"no": "No", "yes": "Yes"}) for i in range(3)}
        text = "- option_0: yes\n2. option_1 - no\n**option_2**: yes"
        self.assertEqual(A.extract(text, o), {"option_0": "yes", "option_1": "no", "option_2": "yes"})

    def test_a_missing_question_is_none_not_a_guess(self) -> None:
        o = {f"option_{i}": P.option_list({"no": "No", "yes": "Yes"}) for i in range(2)}
        self.assertEqual(A.extract("ANSWER option_0: yes", o), {"option_0": "yes", "option_1": None})

    def test_chess_moves_by_uci_or_san(self) -> None:
        crit = {f"{c}2{c}4": {"uci": f"{c}2{c}4", "san": f"{c}4"} for c in "abcdefgh"}
        crit.update({f"{c}7{c}5": {"uci": f"{c}7{c}5", "san": f"{c}5"} for c in "abcdefgh"})
        crit.update({f"{c}2{c}3": {"uci": f"{c}2{c}3", "san": f"{c}3"} for c in "abcdefgh"})
        crit.update({"e7e8q": {"uci": "e7e8q", "san": "e8=Q"}, "e7e8r": {"uci": "e7e8r", "san": "e8=R"}, "g1f3": {"uci": "g1f3", "san": "Nf3"},
                     "f1b5": {"uci": "f1b5", "san": "Bb5+"}, "c1h6": {"uci": "c1h6", "san": "Bh6"}})
        o = {"move": P.option_list(crit)}
        self.assertGreater(len(o["move"]), 26)
        self.assertEqual(A.extract("ANSWER move: g1f3", o)["move"], "g1f3")
        self.assertEqual(A.extract("ANSWER move: Nf3", o)["move"], "g1f3")
        self.assertEqual(A.extract("ANSWER move: **e7e8q**", o)["move"], "e7e8q")
        self.assertEqual(A.extract("ANSWER move: e8=R", o)["move"], "e7e8r")
        self.assertIsNone(A.extract("ANSWER move: e7e8", o)["move"])           # a prefix of a promotion is not a move
        self.assertEqual(A.extract("I like f1b5 but g1f3 is safer. ANSWER move: g1f3", o)["move"], "g1f3")

    def test_duplicate_markers_for_other_questions_are_ignored(self) -> None:
        o = {"q1": opts("AB"), "q2": opts("AB")}
        self.assertEqual(A.extract("ANSWER q1: A\nANSWER q2: B", o), {"q1": "A", "q2": "B"})
        self.assertEqual(A.extract("ANSWER q2: B", o), {"q1": None, "q2": "B"})


class SampleContract(unittest.TestCase):
    @staticmethod
    def cand(cid, group, run_id=None, track="t"):
        rid = run_id or f"{cid}:{group}"
        return S.Candidate(run_id=rid, catalog_id=cid, group_id=group, track=track, payload_sha256="sha-" + rid)

    def population(self, n=60, benches=(25, 31, 58)):
        return [self.cand(b, f"g{i}") for b in benches for i in range(n)]

    def test_rank_hash_format_is_part_of_the_contract(self) -> None:
        import hashlib
        self.assertEqual(S.rank_hash(20261021, "case", 25, "g1"), hashlib.sha256(b"20261021|case|25|g1").hexdigest())

    def test_known_sample_never_changes(self) -> None:
        # Pre-registered: the same seed must pick the same cases forever, on every Python and platform.
        d = S.draw([self.cand(25, f"g{i}") for i in range(30)], per_benchmark=4, benchmarks=(25,))
        self.assertEqual([c.group_id for c in d.chosen[25]], ["g21", "g18", "g24", "g22"])

    def test_deterministic_and_independent_of_input_order(self) -> None:
        pop = self.population()
        a = S.draw(pop, per_benchmark=10, benchmarks=(25, 31, 58))
        b = S.draw(list(reversed(pop)), per_benchmark=10, benchmarks=(25, 31, 58))
        self.assertEqual(a.chosen, b.chosen)
        self.assertEqual(a.total(), 30)

    def test_a_smaller_sample_is_a_prefix_of_a_larger_one(self) -> None:
        pop = self.population()
        small = S.draw(pop, per_benchmark=5, benchmarks=(25, 31, 58)).chosen
        big = S.draw(pop, per_benchmark=12, benchmarks=(25, 31, 58)).chosen
        for b in (25, 31, 58):
            self.assertEqual(big[b][:5], small[b])

    def test_n_per_benchmark_when_the_population_allows_else_all(self) -> None:
        pop = self.population(n=60) + [self.cand(43, f"x{i}") for i in range(3)]
        d = S.draw(pop, per_benchmark=10, benchmarks=(25, 43))
        self.assertEqual({b: len(v) for b, v in d.chosen.items()}, {25: 10, 43: 3})
        self.assertEqual(d.population, {25: 60, 43: 3})

    def test_zero_or_none_means_every_case(self) -> None:
        pop = self.population(n=7)
        for n in (0, None):
            self.assertEqual({b: len(v) for b, v in S.draw(pop, per_benchmark=n, benchmarks=(25, 31)).chosen.items()}, {25: 7, 31: 7})

    def test_seed_changes_the_sample_and_the_default_is_the_pre_registered_one(self) -> None:
        pop = self.population()
        a = S.draw(pop, per_benchmark=10, benchmarks=(25,))
        b = S.draw(pop, per_benchmark=10, seed=1, benchmarks=(25,))
        c = S.draw(pop, per_benchmark=10, seed=C.SEED, benchmarks=(25,))
        self.assertNotEqual(a.chosen[25], b.chosen[25])
        self.assertEqual(a.chosen[25], c.chosen[25])

    def test_one_request_per_case(self) -> None:
        pop = [self.cand(30, f"GSM8K:{i}", run_id=f"v0.3-gsm8k:{i}:{k}") for i in range(20) for k in (4, 10)]
        d = S.draw(pop, per_benchmark=0, benchmarks=(30,))
        self.assertEqual(len(d.chosen[30]), 20)
        self.assertEqual(len({c.group_id for c in d.chosen[30]}), 20)
        again = S.draw(list(reversed(pop)), per_benchmark=0, benchmarks=(30,))
        self.assertEqual(d.chosen, again.chosen)                       # the same of the two rows is picked
        every = S.draw(pop, per_benchmark=0, benchmarks=(30,), one_per_case=False)
        self.assertEqual(len(every.chosen[30]), 40)

    def test_ineligible_cases_are_skipped_and_the_next_ranked_takes_their_place(self) -> None:
        pop = self.population(n=40, benches=(25,))
        base = S.draw(pop, per_benchmark=8, benchmarks=(25,)).chosen[25]
        banned = {base[0].run_id, base[3].run_id}
        d = S.draw(pop, per_benchmark=8, benchmarks=(25,), eligible=lambda c: c.run_id not in banned)
        got = d.chosen[25]
        self.assertEqual(len(got), 8)
        self.assertFalse(banned & {c.run_id for c in got})
        self.assertEqual(got[:2], [base[1], base[2]])                  # same order, ineligible ones removed
        self.assertEqual(d.ineligible[25], 2)

    def test_a_case_with_one_eligible_row_uses_that_row(self) -> None:
        pop = [self.cand(30, "p", run_id="a"), self.cand(30, "p", run_id="b")]
        for only in ("a", "b"):
            d = S.draw(pop, per_benchmark=0, benchmarks=(30,), eligible=lambda c, only=only: c.run_id == only)
            self.assertEqual([c.run_id for c in d.chosen[30]], [only])

    def test_a_benchmark_without_rows_is_reported_not_fatal(self) -> None:
        d = S.draw(self.population(benches=(25,)), per_benchmark=5, benchmarks=(25, 45))
        self.assertEqual(d.missing, [45])
        self.assertEqual(sorted(d.chosen), [25])

    def test_other_benchmarks_are_ignored(self) -> None:
        d = S.draw(self.population(benches=(25, 99)), per_benchmark=5, benchmarks=(25,))
        self.assertEqual(sorted(d.chosen), [25])

    def test_interleave_is_round_robin_by_rank(self) -> None:
        pop = self.population(n=6, benches=(25, 31, 58)) + [self.cand(58, f"extra{i}") for i in range(3)]
        d = S.draw(pop, per_benchmark=4, benchmarks=(25, 31, 58))
        order = S.interleave(d)
        self.assertEqual([c.catalog_id for c in order[:6]], [25, 31, 58, 25, 31, 58])
        self.assertEqual(order[:3], [d.chosen[25][0], d.chosen[31][0], d.chosen[58][0]])
        self.assertEqual(len(order), 12)
        short = S.draw(self.population(n=2, benches=(25,)) + self.population(n=5, benches=(31,)), per_benchmark=0, benchmarks=(25, 31))
        self.assertEqual([c.catalog_id for c in S.interleave(short)], [25, 31, 25, 31, 31, 31, 31])

    @staticmethod
    def row(cid=25, qs=None, expected=None, scoring=None, run_id="25:x"):
        qs = {"q": {"type": "choice", "instructions": "i", "criteria": {"A": "a", "B": "b"}}} if qs is None else qs
        row = {"state": "s", "questions": qs, "expected": {"q": "A"} if expected is None else expected,
               "_evaluation": {"run_id": run_id, "catalog_id": cid, "group_id": "g", "track": "t", "payload_sha256": "abc"}}
        if scoring is not None:
            row["scoring"] = scoring
        return row

    def test_candidate_of_accepts_choice_rows_with_a_gold(self) -> None:
        c = S.candidate_of(self.row())
        self.assertEqual((c.run_id, c.catalog_id, c.group_id, c.track, c.payload_sha256), ("25:x", 25, "g", "t", "abc"))

    def test_candidate_of_refuses_what_the_thinker_cannot_answer(self) -> None:
        noul = {"q": {"type": "noul", "instructions": "i"}}
        self.assertIsNone(S.candidate_of(self.row(qs=noul)))
        self.assertIsNone(S.candidate_of(self.row(expected={"q": None})))
        self.assertIsNone(S.candidate_of(self.row(expected={})))
        self.assertIsNone(S.candidate_of(self.row(qs={})))

    def test_gold_sets(self) -> None:
        self.assertEqual(S.gold_sets(self.row()), {"q": ["A"]})
        chess = self.row(cid=31, scoring={"type": "chess_reference_value", "accepted": ["a1a2", "b1b2"]}, expected={"q": "a1a2"})
        self.assertEqual(S.gold_sets(chess), {"q": ["a1a2", "b1b2"]})
        multi = self.row(qs={f"o{i}": {"type": "choice", "instructions": "i", "criteria": {"no": "No", "yes": "Yes"}} for i in range(2)},
                         expected={"o0": "no", "o1": "yes"}, scoring={"accepted": ["zzz"]})
        self.assertEqual(S.gold_sets(multi), {"o0": ["no"], "o1": ["yes"]})      # `accepted` only widens one-question rows

    def test_scoring_accepted_widens_the_gold_only_of_the_benchmarks_where_the_kit_applies_it(self) -> None:
        self.assertEqual(S.ACCEPTED_BENCHMARKS, (31, 50))
        scoring = {"type": "chess_reference_value", "accepted": ["a1a2", "b1b2"]}
        for cid, expected in ((31, ["a1a2", "b1b2"]), (50, ["a1a2", "b1b2"]), (25, ["a1a2"]), (58, ["a1a2"]), (30, ["a1a2"])):
            self.assertEqual(S.gold_sets(self.row(cid=cid, scoring=scoring, expected={"q": "a1a2"})), {"q": expected}, cid)

    def test_decision_from_answers_keeps_probabilities_and_the_reported_choice(self) -> None:
        qs = {"q": {"type": "choice", "criteria": {"A": "a", "B": "b"}}}
        ok = {"q": {"type": "choice", "choice": "B", "probabilities": {"A": 0.25, "B": 0.75}, "confidence": 0.5}}
        self.assertEqual(S.decision_from_answers(ok, qs), {"q": {"choice": "B", "probs": {"A": 0.25, "B": 0.75}}})

    def test_decision_from_answers_refuses_incomplete_or_odd_answers(self) -> None:
        qs = {"q": {"type": "choice", "criteria": {"A": "a", "B": "b"}}}
        bad = [None, {}, {"q": None}, {"q": {"choice": "A"}}, {"q": {"probabilities": {"A": 1.0}}},
               {"q": {"probabilities": {"A": 0.5, "B": 0.5, "C": 0.0}}}, {"q": {"probabilities": {"A": float("nan"), "B": 0.5}}},
               {"q": {"probabilities": {"A": -0.1, "B": 1.1}}}, {"q": {"probabilities": {"A": "0.5", "B": 0.5}}},
               {"q": {"probabilities": {"A": True, "B": 0.5}}}]
        for answers in bad:
            self.assertIsNone(S.decision_from_answers(answers, qs), answers)


def prow(bench, i, gold="A", keys=("A", "B", "C", "D"), picks=None, accepted=None, qid="q"):
    """A pack row.  `picks` maps a decision model to the key it favors (probability .7, the rest share .3)."""
    decisions = {}
    for m, pick in (picks or {}).items():
        rest = [k for k in keys if k != pick]
        decisions[m] = {qid: {"choice": pick, "probs": {k: (0.7 if k == pick else 0.3 / len(rest)) for k in keys}}}
    return {"id": f"{bench}:{i}", "benchmark": bench, "track": "t", "group": f"{bench}-{i}", "payload_sha256": f"h{bench}-{i}",
            "state": "s", "questions": {qid: {"type": "choice", "instructions": "i", "criteria": {k: k.lower() for k in keys}}},
            "gold": {qid: list(accepted or [gold])}, "decisions": decisions}


def trec(row, pick, status="ok", **extra):
    rec = {"kind": "row", "id": row["id"], "benchmark": row["benchmark"], "status": status,
           "answers": {q: pick for q in row["questions"]}, "tokens": {"completion": 10}, "seconds": 1.0}
    if status == "ok":
        rec["thought"] = True
    rec.update(extra)
    return rec


def official_header(**over):
    """The header of a results file made with the registered protocol: the pinned checkpoint, thinking on, the model card's sampling."""
    return {"kind": "header", "format": C.RESULT_FORMAT, "config": fake_config(**{"model": C.QWEN_MODEL, "revision": C.QWEN_REVISION, **over}), "config_fingerprint": "f"}


def official(rows, recs, **kw):
    """build_report as the gate runs it: the registered bootstrap, seed and target, and the header of a registered run."""
    kw.setdefault("header", official_header())
    kw.setdefault("B", C.BOOTSTRAP_B)
    kw.setdefault("seed", C.SEED)
    return K.build_report(rows, recs, **kw)


def scenario(spec):
    """spec: {benchmark: [(thinker_hit, {model: hit}), ...]} -> (pack rows, thinker records); gold is always A."""
    rows, recs = [], []
    for bench, items in spec.items():
        for i, (t, ms) in enumerate(items):
            row = prow(bench, i, picks={m: ("A" if h else "B") for m, h in ms.items()})
            rows.append(row)
            recs.append(trec(row, "A" if t else "B"))
    return rows, recs


def naive_counts(hits, B, seed):
    """Reference resampler: replicate by replicate, benchmark by benchmark (ascending), n draws of int(random() * n)."""
    rnd = random.Random(seed)
    systems = list(hits)
    benches = sorted(hits[systems[0]])
    out = {s: {b: [] for b in benches} for s in systems}
    for _ in range(B):
        for b in benches:
            n = len(hits[systems[0]][b])
            idx = [int(rnd.random() * n) for _ in range(n)]
            for s in systems:
                out[s][b].append(sum(hits[s][b][i] for i in idx))
    return out


class ScoringContract(unittest.TestCase):
    def test_argmax_takes_the_first_maximum_in_option_order(self) -> None:
        self.assertEqual(K.argmax_key({"A": .4, "B": .4, "C": .2}, ["A", "B", "C"]), "A")
        self.assertEqual(K.argmax_key({"A": .4, "B": .4, "C": .2}, ["B", "A", "C"]), "B")
        self.assertEqual(K.argmax_key({"A": .1, "B": .7, "C": .2}, ["A", "B", "C"]), "B")
        self.assertEqual(K.argmax_key({"x": 0, "y": 0}, ["x", "y"]), "x")

    def test_decision_picks_read_the_probabilities_not_the_reported_choice(self) -> None:
        row = prow(25, 0, picks={"m": "B"})
        row["decisions"]["m"]["q"]["choice"] = "C"
        self.assertEqual(K.decision_picks(row, "m"), {"q": "B"})

    def test_a_missing_probability_is_an_error_not_a_silent_miss(self) -> None:
        row = prow(25, 0, picks={"m": "B"})
        del row["decisions"]["m"]["q"]["probs"]["D"]
        with self.assertRaises(ValueError):
            K.decision_picks(row, "m")

    def test_row_hit_needs_every_question_right(self) -> None:
        self.assertTrue(K.row_hit({"q": ["A"]}, {"q": "A"}))
        self.assertFalse(K.row_hit({"q": ["A"]}, {"q": "B"}))
        self.assertFalse(K.row_hit({"q": ["A"]}, {"q": None}))
        self.assertFalse(K.row_hit({"q": ["A"]}, {}))
        self.assertTrue(K.row_hit({"q": ["a1a2", "b1b2"]}, {"q": "b1b2"}))              # chess ties
        self.assertTrue(K.row_hit({"o1": ["yes"], "o2": ["no"]}, {"o1": "yes", "o2": "no"}))
        self.assertFalse(K.row_hit({"o1": ["yes"], "o2": ["no"]}, {"o1": "yes", "o2": "yes"}))   # SATA: one wrong = case wrong
        self.assertFalse(K.row_hit({}, {}))

    def test_row_chance_is_the_kit_chance_of_the_row(self) -> None:
        self.assertAlmostEqual(K.row_chance(prow(25, 0)), 0.25)
        self.assertAlmostEqual(K.row_chance(prow(31, 0, accepted=["A", "B"])), 0.5)
        sata = {"questions": {f"o{i}": {"criteria": {"no": "No", "yes": "Yes"}} for i in range(3)}, "gold": {f"o{i}": ["yes"] for i in range(3)}}
        self.assertAlmostEqual(K.row_chance(sata), 0.125)

    def test_tables_accuracy_macro_pooled_and_skill(self) -> None:
        rows, recs = scenario({
            25: [(1, {"m1": 1, "m2": 0}), (1, {"m1": 0, "m2": 0}), (0, {"m1": 1, "m2": 1}), (0, {"m1": 0, "m2": 1})],
            58: [(1, {"m1": 1, "m2": 1}), (1, {"m1": 1, "m2": 0})],
        })
        s = K.score_pack(rows, recs)
        self.assertEqual(s.systems, ["thinker", "m1", "m2"])
        self.assertEqual(s.benchmarks, [25, 58])
        self.assertEqual(s.hits["thinker"], {25: [1, 1, 0, 0], 58: [1, 1]})
        self.assertEqual(s.hits["m1"], {25: [1, 0, 1, 0], 58: [1, 1]})
        self.assertEqual(s.hits["m2"], {25: [0, 0, 1, 1], 58: [1, 0]})
        self.assertAlmostEqual(s.acc("thinker", 25), 0.5)
        self.assertAlmostEqual(s.macro_acc("thinker"), 0.75)
        self.assertAlmostEqual(s.pooled_acc("thinker"), 4 / 6)
        self.assertAlmostEqual(s.skill("thinker", 25), C.skill(0.5, 0.25))
        self.assertAlmostEqual(s.macro_skill("thinker"), (C.skill(0.5, 0.25) + C.skill(1.0, C.chance_of(58))) / 2)
        self.assertAlmostEqual(s.macro_acc("m2"), (0.5 + 0.5) / 2)

    def test_unparseable_and_failed_rows_are_wrong_error_rows_are_missing(self) -> None:
        rows, _ = scenario({25: [(1, {"m": 1})] * 4})
        recs = [trec(rows[0], "A"),
                {"kind": "row", "id": rows[1]["id"], "status": "ok", "answers": {"q": None}},        # unparseable: wrong
                {"kind": "row", "id": rows[2]["id"], "status": "failed", "answers": {}},             # terminal failure: wrong
                {"kind": "row", "id": rows[3]["id"], "status": "error", "answers": {}}]              # transient: to be retried
        s = K.score_pack(rows, recs)
        self.assertEqual(s.hits["thinker"], {25: [1, 0, 0]})
        self.assertEqual(s.hits["m"], {25: [1, 1, 1]})        # decision models are scored on the same rows only
        self.assertEqual(s.errors, [rows[3]["id"]])
        self.assertEqual(s.missing, [])

    def test_missing_rows_are_dropped_for_every_system_and_foreign_ids_are_listed(self) -> None:
        rows, recs = scenario({25: [(1, {"m": 1})] * 3, 58: [(0, {"m": 0})] * 2})
        recs = [recs[0], recs[2], recs[3]]                        # 25:1 and 58:1 never ran
        recs.append(trec(prow(25, 99), "A"))                      # 25:99 is not in the pack
        s = K.score_pack(rows, recs)
        self.assertEqual(s.ids, {25: ["25:0", "25:2"], 58: ["58:0"]})
        self.assertEqual(s.hits["m"], {25: [1, 1], 58: [0]})
        self.assertEqual(sorted(s.missing), ["25:1", "58:1"])
        self.assertEqual(s.foreign, ["25:99"])
        self.assertFalse(s.complete)

    def test_a_benchmark_with_no_scored_row_leaves_the_macro(self) -> None:
        rows, recs = scenario({25: [(1, {"m": 1})] * 2, 58: [(1, {"m": 1})] * 2})
        s = K.score_pack(rows, [r for r in recs if r["benchmark"] == 25])
        self.assertEqual(s.benchmarks, [25])
        self.assertAlmostEqual(s.macro_acc("thinker"), 1.0)

    def test_the_last_record_of_an_id_wins(self) -> None:
        rows, recs = scenario({25: [(1, {"m": 1})] * 2})
        s = K.score_pack(rows, [recs[0], recs[1], trec(rows[0], "B")])
        self.assertEqual(s.hits["thinker"], {25: [0, 1]})

    def test_models_are_those_every_row_has_and_can_be_restricted(self) -> None:
        rows, recs = scenario({25: [(1, {"m1": 1, "m2": 1})] * 2})
        del rows[1]["decisions"]["m2"]
        self.assertEqual(K.score_pack(rows, recs).systems, ["thinker", "m1"])
        self.assertEqual(K.score_pack(rows, recs, models=["m1"]).systems, ["thinker", "m1"])
        with self.assertRaises(ValueError):
            K.score_pack(rows, recs, models=["m2"])

    def test_reported_choice_that_disagrees_with_the_probabilities_is_counted(self) -> None:
        rows, recs = scenario({25: [(1, {"m": 1})] * 3})
        rows[1]["decisions"]["m"]["q"]["choice"] = "C"
        self.assertEqual(K.score_pack(rows, recs).choice_mismatch, {"m": 1})

    def test_load_thinker_reads_header_rows_and_ignores_a_cut_last_line(self) -> None:
        import tempfile
        rows, recs = scenario({25: [(1, {"m": 1})] * 2})
        header = {"kind": "header", "format": C.RESULT_FORMAT, "config_fingerprint": "f"}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text("\n".join(json.dumps(x) for x in [header, *recs]) + "\n" + '{"kind":"row","id":"25:9","sta', encoding="utf-8")
            h, got = K.load_thinker(p)
        self.assertEqual(h, header)
        self.assertEqual([r["id"] for r in got], ["25:0", "25:1"])

    def test_thinker_stats(self) -> None:
        rows, _ = scenario({25: [(1, {"m": 1})] * 5})
        recs = [trec(rows[0], "A", tokens={"completion": 100}, seconds=10.0),
                trec(rows[1], "A", tokens={"completion": 300}, seconds=30.0),
                trec(rows[2], "B", tokens={"completion": 200}, seconds=20.0, truncated=True, finish_reason="length"),
                {"kind": "row", "id": rows[3]["id"], "status": "ok", "answers": {"q": None}, "tokens": {"completion": 400}, "seconds": 40.0},
                {"kind": "row", "id": rows[4]["id"], "status": "failed", "answers": {}}]
        st = K.thinker_stats(K.score_pack(rows, recs))
        self.assertEqual((st["rows"], st["failed"], st["unparseable"], st["truncated"]), (5, 1, 1, 1))
        self.assertEqual(st["completion_tokens"]["total"], 1000)
        self.assertEqual(st["completion_tokens"]["median"], 250)
        self.assertEqual(st["completion_tokens"]["max"], 400)
        self.assertEqual(st["seconds"]["total"], 100.0)
        self.assertEqual(st["seconds"]["p80"], 40.0)


class BootstrapContract(unittest.TestCase):
    def test_percentile_ci_uses_fixed_ranks(self) -> None:
        xs = list(range(10_000))
        self.assertEqual(K.percentile_ci(xs, 0.95), (250, 9749))
        self.assertEqual(K.percentile_ci(list(reversed(xs)), 0.95), (250, 9749))
        self.assertEqual(K.percentile_ci(list(range(200)), 0.95), (5, 194))
        # a*B is not an integer: the lower rank rounds DOWN (2.5 -> 2), the upper one is ceil(97.5) - 1 = 97
        self.assertEqual(K.percentile_ci(list(range(100)), 0.95), (2, 97))
        self.assertEqual(K.percentile_ci(list(range(30)), 0.95), (0, 29))
        self.assertEqual(K.percentile_ci([3.0], 0.95), (3.0, 3.0))
        with self.assertRaises(ValueError):
            K.percentile_ci([], 0.95)

    @staticmethod
    def random_scenario(n25=12, n58=7, seed=3):
        rnd = random.Random(seed)
        spec = {25: [(rnd.random() < .6, {"m1": rnd.random() < .5, "m2": rnd.random() < .4}) for _ in range(n25)],
                58: [(rnd.random() < .7, {"m1": rnd.random() < .6, "m2": rnd.random() < .5}) for _ in range(n58)]}
        return scenario(spec)

    def test_counts_equal_the_naive_resampler_with_the_documented_draw_order(self) -> None:
        s = K.score_pack(*self.random_scenario())
        rep = K.bootstrap(s.hits, B=60, seed=7)
        naive = naive_counts(s.hits, 60, 7)
        for sysname in s.systems:
            for b in s.benchmarks:
                self.assertEqual(list(rep.counts[sysname][b]), naive[sysname][b], (sysname, b))

    def test_the_order_of_the_systems_does_not_change_the_draws(self) -> None:
        s = K.score_pack(*self.random_scenario())
        flipped = {k: s.hits[k] for k in reversed(list(s.hits))}
        a = K.bootstrap(s.hits, B=40, seed=11)
        b = K.bootstrap(flipped, B=40, seed=11)
        for sysname in s.systems:
            for bench in s.benchmarks:
                self.assertEqual(list(a.counts[sysname][bench]), list(b.counts[sysname][bench]))

    def test_large_samples_do_not_overflow_the_packed_counts(self) -> None:
        n = 70_000
        hits = {"thinker": {1: [1] * n}, "m": {1: [0] * (n // 2) + [1] * (n // 2)}}
        rep = K.bootstrap(hits, B=2, seed=1)
        self.assertEqual(list(rep.counts["thinker"][1]), [n, n])
        self.assertTrue(all(abs(c - n // 2) < 2000 for c in rep.counts["m"][1]))

    def test_seed_is_reproducible_and_matters(self) -> None:
        s = K.score_pack(*self.random_scenario())
        a = K.bootstrap(s.hits, B=80, seed=1)
        self.assertEqual(a.delta_pts("thinker", "m1"), K.bootstrap(s.hits, B=80, seed=1).delta_pts("thinker", "m1"))
        self.assertNotEqual(a.delta_pts("thinker", "m1"), K.bootstrap(s.hits, B=80, seed=2).delta_pts("thinker", "m1"))

    def test_same_rows_same_outcomes_gives_a_zero_delta_in_every_replicate(self) -> None:
        s = K.score_pack(*scenario({25: [(i % 2 == 0, {"m": i % 2 == 0}) for i in range(30)]}))
        rep = K.bootstrap(s.hits, B=100, seed=5)
        self.assertEqual(set(rep.delta_pts("thinker", "m")), {0.0})

    def test_thinker_always_right_model_always_wrong_gives_plus_100(self) -> None:
        s = K.score_pack(*scenario({25: [(1, {"m": 0})] * 40}))
        rep = K.bootstrap(s.hits, B=100, seed=5)
        self.assertEqual(K.percentile_ci(rep.delta_pts("thinker", "m")), (100.0, 100.0))

    def test_resampling_is_stratified_by_benchmark(self) -> None:
        # Benchmark 25: thinker wins every row.  Benchmark 58: model wins every row.  With each
        # benchmark resampled inside itself the macro delta is exactly 0 in every replicate; a pooled
        # resample would let the mix of the two benchmarks move it.
        s = K.score_pack(*scenario({25: [(1, {"m": 0})] * 10, 58: [(0, {"m": 1})] * 30}))
        rep = K.bootstrap(s.hits, B=200, seed=9)
        self.assertEqual(set(rep.delta_pts("thinker", "m")), {0.0})

    def test_ci_matches_the_normal_approximation_on_a_large_paired_sample(self) -> None:
        # n=400: thinker-only right on 60 rows, model-only right on 20, both right on 160, neither on 160.
        # delta = (60 - 20) / 400 = +10.0 pts; SE of the paired difference = sqrt((0.2 - 0.1 ** 2) / 400) = 2.18 pts.
        items = [(1, {"m": 0})] * 60 + [(0, {"m": 1})] * 20 + [(1, {"m": 1})] * 160 + [(0, {"m": 0})] * 160
        s = K.score_pack(*scenario({25: items}))
        rep = K.bootstrap(s.hits, B=C.BOOTSTRAP_B, seed=C.SEED)
        lo, hi = K.percentile_ci(rep.delta_pts("thinker", "m"))
        self.assertAlmostEqual(100 * (s.macro_acc("thinker") - s.macro_acc("m")), 10.0)
        self.assertTrue(5.0 < lo < 6.5, lo)
        self.assertTrue(13.5 < hi < 15.0, hi)

    def test_macro_series_and_per_benchmark_series(self) -> None:
        s = K.score_pack(*self.random_scenario())
        rep = K.bootstrap(s.hits, B=30, seed=4)
        naive = naive_counts(s.hits, 30, 4)
        want = [(naive["thinker"][25][r] / 12 + naive["thinker"][58][r] / 7) / 2 for r in range(30)]
        for got, w in zip(rep.macro("thinker"), want):
            self.assertAlmostEqual(got, w)
        self.assertEqual(rep.bench("m1", 58), [c / 7 for c in naive["m1"][58]])


class VerdictContract(unittest.TestCase):
    def test_pass_needs_ten_points_and_a_positive_ci_floor(self) -> None:
        self.assertEqual(K.decide(10.0, 0.01, 20.0)["status"], "PASS")
        self.assertEqual(K.decide(25.0, 12.0, 38.0)["status"], "PASS")

    def test_delta_below_the_target_fails(self) -> None:
        v = K.decide(9.99, 5.0, 20.0)
        self.assertEqual((v["status"], v["reasons"]), ("FAIL", ["delta_below_min"]))

    def test_a_ci_floor_of_zero_or_less_fails(self) -> None:
        for lo in (0.0, -0.5):
            v = K.decide(12.0, lo, 24.0)
            self.assertEqual((v["status"], v["reasons"]), ("FAIL", ["ci_low_not_above_zero"]), lo)

    def test_both_conditions_can_fail_together(self) -> None:
        self.assertEqual(K.decide(2.0, -3.0, 7.0)["reasons"], ["delta_below_min", "ci_low_not_above_zero"])

    def test_float_noise_does_not_flip_a_boundary(self) -> None:
        self.assertLess(100 * (0.6 - 0.5), 10.0)                     # the float really is below 10
        self.assertEqual(K.decide(100 * (0.6 - 0.5), 1.0, 20.0)["status"], "PASS")
        self.assertEqual(K.decide(15.0, 1e-12, 20.0)["status"], "FAIL")      # a floor of 1e-12 is a floor of 0

    def test_custom_target(self) -> None:
        self.assertEqual(K.decide(6.0, 1.0, 11.0, min_delta=5.0)["status"], "PASS")
        self.assertEqual(K.decide(6.0, 1.0, 11.0)["status"], "FAIL")

    def test_a_fail_says_whether_more_rows_could_still_pass(self) -> None:
        self.assertTrue(K.decide(8.0, 3.0, 14.0)["can_still_pass"])          # the CI still holds +10
        self.assertFalse(K.decide(2.0, -3.0, 7.0)["can_still_pass"])         # the CI rules +10 out
        self.assertFalse(K.decide(12.0, 3.0, 20.0)["can_still_pass"])        # a pass has nothing to still pass

    def test_can_still_pass_holds_when_the_ci_ceiling_is_exactly_the_target(self) -> None:
        self.assertTrue(K.decide(4.0, -1.0, 10.0)["can_still_pass"])
        self.assertFalse(K.decide(4.0, -1.0, 9.999)["can_still_pass"])
        self.assertFalse(K.decide(12.0, 3.0, 20.0)["can_still_pass"])             # a PASS has nothing left to pass

    def test_reference_is_the_best_model_by_macro_accuracy_ties_by_name(self) -> None:
        rows, recs = scenario({25: [(1, {"b": 1, "a": 1, "c": 0}), (1, {"b": 0, "a": 0, "c": 1})]})
        s = K.score_pack(rows, recs)
        self.assertEqual(K.pick_reference(s), "a")
        rows, recs = scenario({25: [(1, {"b": 1, "a": 0, "c": 0}), (1, {"b": 1, "a": 0, "c": 1})]})
        self.assertEqual(K.pick_reference(K.score_pack(rows, recs)), "b")


class ReportContract(unittest.TestCase):
    def passing(self):
        # 40 rows: thinker right on 36, best model (m1) on 20, the other model (m2) on 10.
        items = []
        for i in range(40):
            items.append((i < 36, {"m1": i % 2 == 0, "m2": i % 4 == 0}))
        return scenario({25: items})

    def test_a_clear_win_passes_with_the_reference_named(self) -> None:
        rep = official(*self.passing())
        self.assertEqual(rep["verdict"]["status"], "PASS")
        self.assertEqual(rep["reference"], "m1")
        self.assertAlmostEqual(rep["verdict"]["delta_pts"], 40.0)
        self.assertGreater(rep["verdict"]["ci"][0], 10.0)
        self.assertTrue(rep["coverage"]["complete"])
        line = K.verdict_line(rep)
        self.assertTrue(line.startswith("GT VERDICT: PASS"), line)
        self.assertIn("m1", line)
        self.assertIn("+40.0", line)

    def test_a_tie_with_the_best_model_fails_and_says_so(self) -> None:
        rows, _ = self.passing()
        recs = [trec(r, "A" if i % 2 == 0 else "B") for i, r in enumerate(rows)]      # thinker == m1
        rep = official(rows, recs)
        self.assertEqual(rep["verdict"]["status"], "FAIL")
        self.assertIn("delta_below_min", rep["verdict"]["reasons"])
        self.assertAlmostEqual(rep["verdict"]["delta_pts"], 0.0)
        self.assertTrue(K.verdict_line(rep).startswith("GT VERDICT: FAIL"))

    def test_partial_runs_are_labeled(self) -> None:
        rows, recs = self.passing()
        rep = official(rows, recs[:30])
        self.assertFalse(rep["coverage"]["complete"])
        self.assertEqual((rep["coverage"]["rows_scored"], rep["coverage"]["rows_in_pack"]), (30, 40))
        self.assertIn("PARTIAL", K.verdict_line(rep))

    def test_no_scored_row_is_inconclusive_not_a_crash(self) -> None:
        rows, _ = self.passing()
        rep = official(rows, [])
        self.assertEqual(rep["verdict"]["status"], "INCONCLUSIVE")
        self.assertTrue(K.verdict_line(rep).startswith("GT VERDICT: INCONCLUSIVE"))

    def test_report_holds_every_system_and_benchmark(self) -> None:
        rep = official(*self.passing())
        self.assertEqual(sorted(rep["systems"]), ["m1", "m2", "thinker"])
        b = rep["benchmarks"]["25"]
        self.assertEqual((b["name"], b["n"], b["chance"]), ("GPQA Diamond", 40, 0.25))
        self.assertAlmostEqual(b["systems"]["thinker"]["acc"], 0.9)
        self.assertAlmostEqual(b["systems"]["thinker"]["skill"], C.skill(0.9, 0.25))
        self.assertAlmostEqual(b["vs_reference"]["delta_pts"], 40.0)
        self.assertEqual(sorted(rep["deltas"]), ["m1", "m2"])
        self.assertAlmostEqual(rep["deltas"]["m2"]["delta_pts"], 100 * (0.9 - 0.25))
        self.assertEqual(rep["discordant"], {"thinker_only": 18, "reference_only": 2, "both": 18, "neither": 2})

    def test_the_resampled_reference_ci_is_never_tighter_at_the_floor(self) -> None:
        rep = official(*self.passing())
        self.assertLessEqual(rep["verdict"]["ci_reselected"][0], rep["verdict"]["ci"][0] + 1e-9)

    def test_markdown_is_deterministic_and_carries_the_verdict(self) -> None:
        rep = official(*self.passing())
        md = K.render_markdown(rep)
        self.assertEqual(md, K.render_markdown(rep))
        self.assertIn(K.verdict_line(rep), md)
        self.assertIn("GPQA Diamond", md)
        self.assertIn("m2", md)


class VerdictStatusContract(unittest.TestCase):
    """Only a complete run of the registered protocol, with a thinker that was seen thinking, prints PASS or FAIL
    (review of #78, blockers 1 and 2).  Anything else says what it is, in the report and on the last line."""

    TEN = [b for b in C.DEFAULT_BENCHMARKS if b != 45]              # the ten benchmarks of the gate (HLE is gated out)

    def full(self, hit=lambda b, i: True, ref=lambda b, i: i % 2 == 0, per=40, benches=None):
        spec = {b: [(hit(b, i), {"m1": ref(b, i), "m2": i % 4 == 0}) for i in range(per)] for b in (benches or self.TEN)}
        return scenario(spec)

    def assert_not_official(self, rep, status):
        v = rep["verdict"]
        self.assertEqual(v["status"], status)
        line = K.verdict_line(rep)
        self.assertTrue(line.startswith(f"GT VERDICT: {status}"), line)
        self.assertNotIn("GT VERDICT: PASS", line)
        self.assertNotIn("GT VERDICT: FAIL", line)
        self.assertIn(line, K.render_markdown(rep))
        return line

    # -- blocker 1: the two probes of the review
    def test_the_probe_of_three_rows_is_partial_not_pass(self) -> None:
        rows, recs = self.full()
        rep = official(rows, [recs[1], recs[41], recs[81]])                # one row per benchmark, each one the reference misses
        line = self.assert_not_official(rep, "PARTIAL")
        v = rep["verdict"]
        self.assertEqual(v["outcome"], "PASS", "the provisional delta and CI are still there to read")
        self.assertGreater(v["delta_pts"], 10.0)
        self.assertEqual(len(v["ci"]), 2)
        self.assertIn("3 of 400 rows", line)
        self.assertIn("397 not run", line)
        self.assertIn("provisional", line)
        self.assertFalse(rep["coverage"]["complete"])

    def test_transient_errors_on_the_hardest_rows_cannot_turn_a_fail_into_a_pass(self) -> None:
        # 4 benchmarks x 40 rows, per benchmark: both right on 21, only the thinker on 6, only the reference on 3, neither on 10.
        # Thinker 27/40, reference 24/40: +7.5, FAIL.  Drop the 10 rows neither gets right (a transient error on them) and the same run reads +10.0.
        def build(status):
            spec = {b: [(True, {"m1": True})] * 21 + [(True, {"m1": False})] * 6 + [(False, {"m1": True})] * 3 + [(False, {"m1": False})] * 10
                    for b in (25, 28, 31, 57)}
            rows, recs = scenario(spec)
            for r, rec in zip(rows, recs):
                if r["gold"]["q"] == ["A"] and rec["answers"]["q"] != "A" and r["decisions"]["m1"]["q"]["choice"] != "A":
                    rec["status"] = status
            return rows, recs
        rows, recs = build("ok")
        honest = official(rows, recs)
        self.assertEqual((honest["verdict"]["status"], round(honest["verdict"]["delta_pts"], 6)), ("FAIL", 7.5))
        rows, recs = build("error")
        self.assertEqual(len([r for r in recs if r["status"] == "error"]), 40)
        rep = official(rows, recs)
        self.assertEqual(rep["verdict"]["status"], "PARTIAL")
        self.assertEqual(rep["verdict"]["outcome"], "PASS", "this is the bias: on the rows that finished the delta reads +10")
        self.assertAlmostEqual(rep["verdict"]["delta_pts"], 10.0)
        self.assertGreater(rep["verdict"]["ci"][0], 0.0)
        self.assertIn("40 with a transient error", self.assert_not_official(rep, "PARTIAL"))
        rows, recs = build("failed")                                   # a row that was given up is a wrong answer: nothing is dropped
        rep = official(rows, recs)
        self.assertEqual((rep["verdict"]["status"], round(rep["verdict"]["delta_pts"], 6)), ("FAIL", 7.5))
        self.assertTrue(rep["coverage"]["complete"])

    def test_a_partial_run_that_is_also_outside_the_protocol_says_both(self) -> None:
        rows, recs = self.full(benches=[25, 28])
        rep = official(rows, recs[:30], B=500)
        line = self.assert_not_official(rep, "PARTIAL")
        self.assertEqual(rep["verdict"]["exploratory"], ["bootstrap_b"])
        self.assertIn("bootstrap_b", line)

    def test_a_complete_registered_run_keeps_the_plain_verdict_line(self) -> None:
        rows, recs = self.full(benches=[25, 28])
        rep = official(rows, recs)
        self.assertIn(rep["verdict"]["status"], ("PASS", "FAIL"))
        self.assertEqual(rep["verdict"]["outcome"], rep["verdict"]["status"])
        self.assertEqual((rep["verdict"]["exploratory"], rep["verdict"]["inconclusive"], rep["verdict"]["incomplete"]), ([], [], None))
        line = K.verdict_line(rep)
        self.assertTrue(line.startswith(f"GT VERDICT: {rep['verdict']['status']}"))
        for word in ("PARTIAL", "EXPLORATORY", "provisional"):
            self.assertNotIn(word, line)

    # -- blocker 1: a run outside the registered protocol
    def test_every_change_to_the_registered_comparison_is_exploratory_and_named(self) -> None:
        rows, recs = self.full(benches=[25, 28])
        cases = {"bootstrap_b": {"B": 500}, "seed": {"seed": 7}, "min_delta": {"min_delta": 5.0}, "models_subset": {"models": ["m1"]},
                 "pack_mismatch_ignored": {"pack_issues": ["pack_mismatch_ignored"]}, "pack_not_manifest": {"pack_issues": ["pack_not_manifest"]},
                 "manifest_not_checked": {"pack_issues": ["manifest_not_checked"]}}
        for code, kw in cases.items():
            rep = official(rows, recs, **kw)
            line = self.assert_not_official(rep, "EXPLORATORY")
            self.assertEqual(rep["verdict"]["reasons"], [code], code)
            self.assertIn(code, line)
            self.assertIn(rep["verdict"]["outcome"], ("PASS", "FAIL"), code)
            self.assertTrue(rep["verdict"]["exploratory"] == [code], code)

    def test_naming_every_model_is_not_a_subset(self) -> None:
        rows, recs = self.full(benches=[25])
        self.assertIn(official(rows, recs, models=["m2", "m1"])["verdict"]["status"], ("PASS", "FAIL"))

    def test_the_thinker_side_of_the_protocol_is_checked_from_the_header(self) -> None:
        rows, recs = self.full(benches=[25, 28])
        cases = {
            "thinker_config_unknown": {"header": None},
            "thinking_off": {"header": official_header(chat_template_kwargs={"enable_thinking": False})},
            "model_not_pinned": {"header": official_header(model="other/model")},
            "revision_not_declared": {"header": official_header(revision=None)},
            "standard_only": {"header": official_header(standard_only=True)},
            "sampling_not_registered": {"header": official_header(sampling={**C.THINKING_SAMPLING, "temperature": 0.6})},
            "reasoning_effort": {"header": official_header(reasoning_effort="low")},
            "extra_body_overrides_protocol": {"header": official_header(extra_body={"temperature": 0})},
        }
        for code, kw in cases.items():
            rep = official(rows, recs, **kw)
            self.assert_not_official(rep, "EXPLORATORY")
            self.assertEqual(rep["verdict"]["reasons"], [code], code)
        for ok in ({"header": official_header(model=C.QWEN_FP8_MODEL, revision=C.QWEN_FP8_REVISION)},
                   {"header": official_header(quantization="nf4-fp16", backend="transformers")},        # one-sided, but a registered run
                   {"header": official_header(reasoning_effort="xhigh")},
                   {"header": official_header(extra_body={"provider": {"order": ["a"]}})}):
            self.assertIn(official(rows, recs, **ok)["verdict"]["status"], ("PASS", "FAIL"), ok)

    def test_a_reply_without_usage_cannot_be_measured_and_the_run_is_exploratory(self) -> None:
        rows, recs = self.full(benches=[25, 28])
        for rec in recs:
            rec["tokens"] = {"completion": None}
        rep = official(rows, recs)
        self.assert_not_official(rep, "EXPLORATORY")
        self.assertEqual(rep["verdict"]["reasons"], ["usage_missing"])
        self.assertEqual(rep["thinker_stats"]["usage_missing"], 80)
        self.assertIn("usage", K.render_markdown(rep))
        for rec in recs[:50]:                                                          # a few rows without usage are noise
            rec["tokens"] = {"completion": 10}
        self.assertIn(official(rows, recs)["verdict"]["status"], ("PASS", "FAIL"))
        for rec in recs[:40]:                                                          # 40 of 80 without usage: exactly half still counts
            rec["tokens"] = {"completion": None}
        for rec in recs[40:]:
            rec["tokens"] = {"completion": 10}
        self.assertIn(official(rows, recs)["verdict"]["status"], ("PASS", "FAIL"), "exactly half of the replies without usage")
        recs[40]["tokens"] = {"completion": None}
        self.assertEqual(official(rows, recs)["verdict"]["reasons"], ["usage_missing"], "one more than half")

    # -- blocker 2: a thinker that was not seen thinking
    def test_a_thinker_that_did_not_think_is_inconclusive_and_the_line_says_why(self) -> None:
        rows, recs = self.full(benches=[25, 28])
        for rec in recs:
            rec["thought"] = False
        rep = official(rows, recs)
        line = self.assert_not_official(rep, "INCONCLUSIVE")
        self.assertEqual(rep["verdict"]["reasons"], ["thinking_not_detected"])
        self.assertEqual(rep["verdict"]["inconclusive"], ["thinking_not_detected"])
        self.assertIn("thinking not detected", line)
        self.assertIn("0 of 80", line)
        self.assertEqual(rep["thinker_stats"]["thought"], 0)
        self.assertEqual(rep["verdict"]["outcome"], "PASS", "the numbers are still there to read")

    def test_thinking_must_show_in_at_least_half_of_the_rows_that_answered(self) -> None:
        rows, recs = self.full(benches=[25])
        for k, rec in enumerate(recs):
            rec["thought"] = k < 20
        self.assertIn(official(rows, recs)["verdict"]["status"], ("PASS", "FAIL"))
        recs[19]["thought"] = False
        self.assertEqual(official(rows, recs)["verdict"]["status"], "INCONCLUSIVE")

    def test_a_run_in_which_nobody_answered_is_inconclusive_not_a_zero(self) -> None:
        rows, recs = self.full(benches=[25])
        for rec in recs:
            rec["status"], rec["answers"] = "failed", {}
            rec.pop("thought")
        rep = official(rows, recs)
        self.assert_not_official(rep, "INCONCLUSIVE")
        self.assertEqual(rep["verdict"]["reasons"], ["no_replies"])
        self.assertIn("no scored row got a reply", K.verdict_line(rep))

    def test_a_results_file_without_a_header_is_read_as_a_thinking_run_so_a_thinker_never_seen_thinking_is_inconclusive(self) -> None:
        rows, recs = self.full(benches=[25])
        for rec in recs:
            rec["thought"] = False
        rep = official(rows, recs, header=None)
        self.assertEqual(rep["verdict"]["status"], "INCONCLUSIVE")
        self.assertEqual(rep["verdict"]["reasons"], ["thinking_not_detected"])
        self.assertEqual(rep["verdict"]["exploratory"], ["thinker_config_unknown"])

    def test_a_row_without_the_flag_does_not_count_as_thinking(self) -> None:
        rows, recs = self.full(benches=[25])
        for rec in recs:
            del rec["thought"]
        self.assertEqual(official(rows, recs)["verdict"]["reasons"], ["thinking_not_detected"])

    def test_rows_that_did_not_answer_are_not_rows_that_did_not_think(self) -> None:
        rows, recs = self.full(benches=[25])
        for rec in recs[:30]:
            rec["status"], rec["answers"] = "failed", {}
            rec.pop("thought")
        rep = official(rows, recs)
        self.assertIn(rep["verdict"]["status"], ("PASS", "FAIL"))
        self.assertEqual((rep["thinker_stats"]["ok"], rep["thinker_stats"]["thought"]), (10, 10))

    def test_a_thinking_off_control_is_exploratory_not_inconclusive(self) -> None:
        rows, recs = self.full(benches=[25])
        for rec in recs:
            rec["thought"] = False
        rep = official(rows, recs, header=official_header(chat_template_kwargs={"enable_thinking": False}))
        self.assertEqual((rep["verdict"]["status"], rep["verdict"]["reasons"]), ("EXPLORATORY", ["thinking_off"]))

    def test_inconclusive_beats_partial_beats_exploratory(self) -> None:
        rows, recs = self.full(benches=[25, 28])
        for rec in recs:
            rec["thought"] = False
        rep = official(rows, recs[:30], B=500)
        self.assertEqual(rep["verdict"]["status"], "INCONCLUSIVE")
        self.assertEqual(rep["verdict"]["incomplete"], {"missing": 50, "errors": 0})
        self.assertEqual(rep["verdict"]["exploratory"], ["bootstrap_b"])
        line = K.verdict_line(rep)
        self.assertIn("PARTIAL", line)
        self.assertIn("bootstrap_b", line)

    def test_no_decision_model_and_no_scored_row_stay_inconclusive_with_their_old_reasons(self) -> None:
        rows, recs = self.full(benches=[25])
        self.assertEqual(official(rows, [])["verdict"]["reasons"], ["no_scored_rows"])
        for r in rows:
            r["decisions"] = {}
        self.assertEqual(official(rows, recs)["verdict"]["reasons"], ["no_decision_models"])


class ReportWarningsContract(unittest.TestCase):
    """What the report must keep saying: the markdown is what people read, and `hosted` and the Modal app only show the last line."""

    def two(self):
        return scenario({b: [(i % 3 != 0, {"m1": i % 2 == 0}) for i in range(40)] for b in (25, 28)})

    def md(self, mutate=lambda recs: None, **kw):
        rows, recs = self.two()
        mutate(recs)
        rep = official(rows, recs, **kw)
        return rep, K.render_markdown(rep)

    def test_a_run_that_is_not_the_gate_says_so_under_the_verdict_line_and_names_each_reason(self) -> None:
        rep, md = self.md(B=500)
        self.assertIn("NAO e o veredito do gate", md)
        self.assertIn("bootstrap_b", md)
        rep, md = self.md(lambda recs: recs.pop())
        self.assertIn("NAO e o veredito do gate", md)
        self.assertIn("1 linhas sem resultado", md)
        self.assertIn("PARTIAL", md)

    def test_a_run_of_the_registered_protocol_does_not_say_it(self) -> None:
        _, md = self.md()
        self.assertNotIn("NAO e o veredito", md)

    def test_an_inconclusive_run_still_shows_its_numbers(self) -> None:
        def no_thought(recs):
            for r in recs:
                r["thought"] = False
        rep, md = self.md(no_thought)
        self.assertEqual(rep["verdict"]["status"], "INCONCLUSIVE")
        self.assertIn("Acuracia por benchmark", md)
        self.assertIn("thinking_not_detected", md)
        self.assertIn("nao mostram pensamento", md)

    def test_rows_that_did_not_show_thinking_are_counted_even_when_most_did(self) -> None:
        def some(recs):
            for r in recs[:10]:
                r["thought"] = False
        rep, md = self.md(some)
        self.assertIn(rep["verdict"]["status"], ("PASS", "FAIL"))
        self.assertIn("AVISO: 10 de 80 linhas nao mostram pensamento", md)
        def one(recs):
            recs[0]["thought"] = False
        self.assertIn("AVISO: 1 de 80 linhas nao mostram pensamento", self.md(one)[1])
        self.assertNotIn("nao mostram pensamento", self.md()[1])

    def test_standard_only_and_missing_usage_are_in_the_report(self) -> None:
        rep, md = self.md(header=official_header(standard_only=True))
        self.assertIn("- AVISO: `--standard-only` tirou top_k, min_p, repetition_penalty e chat_template_kwargs do pedido", md)
        self.assertIn("`standard_only`: --standard-only", md, "and the reason code is explained under the verdict line")
        self.assertEqual(rep["verdict"]["reasons"], ["standard_only"])

        def no_usage(recs):
            for r in recs:
                r["tokens"] = {"completion": None}
        rep, md = self.md(no_usage)
        self.assertIn("AVISO: 80 de 80 linhas sem contagem de tokens", md)
        self.assertIn("usage_missing", md)

    def test_rows_that_stopped_without_closing_the_thinking_are_counted(self) -> None:
        def cut(recs):
            for r in recs[:4]:
                r["no_close"] = True
                r["answers"] = {"q": None}
        _, md = self.md(cut)
        self.assertIn("AVISO: 4 linhas pararam sem fechar o pensamento", md)

    def test_rows_read_by_a_fallback_are_counted(self) -> None:
        def fall(recs):
            for k, r in enumerate(recs):
                r["read_by"] = {"q": "generic" if k < 7 else "marker"}
        rep, md = self.md(fall)
        self.assertEqual(rep["thinker_stats"]["read_by_fallback"], 7)
        self.assertIn("7 linhas foram lidas sem a linha ANSWER", md)
        for how, counted in (("marker", 0), (None, 0), ("line", 1), ("generic", 1), ("boxed", 1), ("bare", 1)):
            with self.subTest(how=how):
                def one(recs, how=how):
                    for r in recs:
                        r["read_by"] = {"q": "marker"}
                    recs[0]["read_by"] = {"q": how}
                self.assertEqual(self.md(one)[0]["thinker_stats"]["read_by_fallback"], counted)

    def test_rows_given_up_are_counted_as_errors_of_the_model(self) -> None:
        def give_up(recs):
            for r in recs[:3]:
                r.update(status="failed", gave_up=3, answers={})
        rep, md = self.md(give_up)
        self.assertEqual(rep["thinker_stats"]["gave_up"], 3)
        self.assertEqual(rep["thinker_stats"]["failed"], 3)
        self.assertIn("3 linhas desistidas", md)

    def test_a_file_that_went_through_two_endpoints_says_so(self) -> None:
        def mixed(recs):
            for k, r in enumerate(recs):
                r["endpoint"] = "https://a.example/v1" if k < 40 else "https://b.example/v1"
        rep, md = self.md(mixed)
        self.assertEqual(rep["thinker_stats"]["endpoints"], ["https://a.example/v1", "https://b.example/v1"])
        self.assertIn("AVISO: as linhas vieram de 2 endpoints", md)
        self.assertIn("https://b.example/v1", md)
        self.assertIn("quantizacao", md)

        def one(recs):
            for r in recs:
                r["endpoint"] = "https://a.example/v1"
        self.assertNotIn("endpoints", self.md(one)[1])


def q_options(row):
    return P.build_prompt({"state": row["state"], "questions": row["questions"]}).options


def fake_config(**over):
    cfg = {"backend": "fake", "model": "m", "revision": None, "prompt_version": C.PROMPT_VERSION,
           "sampling": dict(C.THINKING_SAMPLING), "chat_template_kwargs": {"enable_thinking": True},
           "reasoning_effort": None, "extra_body": {}, "quantization": None, "dtype": None, "max_tokens": 1000, "seed": None, "endpoint": None}
    cfg.update(over)
    return cfg


class InterpretContract(unittest.TestCase):
    def setUp(self) -> None:
        self.row = prow(25, 0)
        self.options = q_options(self.row)

    def parsed(self, thinking=False, **gen):
        return T.interpret(T.GenResult(**gen), self.options, thinking=thinking)

    def test_reasoning_split_by_the_server_and_a_reply_with_the_answer(self) -> None:
        p = self.parsed(reasoning="maybe C, no, B", content="ANSWER q: B", finish_reason="stop")
        self.assertEqual((p.answers, p.reasoning, p.final, p.truncated), ({"q": "B"}, "maybe C, no, B", "ANSWER q: B", False))

    def test_reasoning_inline_with_think_tags(self) -> None:
        p = self.parsed(content="it is A or D...\n</think>\n\nANSWER q: D", finish_reason="stop")
        self.assertEqual(p.answers, {"q": "D"})
        self.assertIn("A or D", p.reasoning)

    def test_cut_while_thinking_is_never_read_as_an_answer(self) -> None:
        # The template opens <think> itself, so a cut reply has no tag at all; its text is reasoning.
        p = self.parsed(content="so the answer is probably B because... hmm, or C", finish_reason="length")
        self.assertEqual(p.answers, {"q": None})
        self.assertIsNone(p.final)
        self.assertTrue(p.truncated)

    def test_cut_while_thinking_with_the_reasoning_apart(self) -> None:
        p = self.parsed(reasoning="long thinking about B and C", content="", finish_reason="length")
        self.assertEqual((p.answers, p.truncated), ({"q": None}, True))

    def test_a_service_that_hides_the_reasoning_returns_just_the_reply(self) -> None:
        p = self.parsed(content="ANSWER q: C", finish_reason="stop")
        self.assertEqual((p.answers, p.truncated), ({"q": "C"}, False))

    def test_a_thinking_run_does_not_read_a_stop_without_a_closing_tag_as_an_answer(self) -> None:
        # The template opens <think> itself: a reply that stopped with no </think> and no reasoning field is thinking that never ended
        # (or an endpoint that ignored the request for it), and its text must not be scanned for an answer.
        text = "Maybe the answer is B... or no, C.\nANSWER q: B"
        p = self.parsed(thinking=True, content=text, finish_reason="stop")
        self.assertEqual((p.answers, p.final, p.no_close), ({"q": None}, None, True))
        for kw in ({"reasoning": "server side thinking"}, {"reasoning_tokens": 300}):
            p = self.parsed(thinking=True, content="ANSWER q: B", finish_reason="stop", **kw)
            self.assertEqual((p.answers, p.no_close), ({"q": "B"}, False), kw)
        p = self.parsed(thinking=True, content="t\n</think>\nANSWER q: B", finish_reason="stop")
        self.assertEqual((p.answers, p.no_close), ({"q": "B"}, False))
        p = self.parsed(thinking=False, content=text, finish_reason="stop")                  # a control without thinking is read whole
        self.assertEqual((p.answers, p.no_close), ({"q": "B"}, False))

    def test_a_cut_reply_is_truncated_not_no_close(self) -> None:
        p = self.parsed(thinking=True, content="so the answer is B", finish_reason="length")
        self.assertEqual((p.truncated, p.no_close), (True, False))

    def test_cut_inside_the_reply_after_the_thinking_still_reads_a_written_answer(self) -> None:
        p = self.parsed(content="reasoning</think>\n\nANSWER q: A\nbecause the", finish_reason="length")
        self.assertEqual((p.answers, p.truncated), ({"q": "A"}, True))

    def test_unreadable_reply_is_none(self) -> None:
        self.assertEqual(self.parsed(content="x</think>\n\nI cannot decide.", finish_reason="stop").answers, {"q": None})
        self.assertEqual(self.parsed(content=None, finish_reason="stop").answers, {"q": None})

    def test_many_questions(self) -> None:
        row = prow(33, 0)
        row["questions"] = {f"o{i}": {"type": "choice", "instructions": "i", "criteria": {"no": "No", "yes": "Yes"}} for i in range(3)}
        p = T.interpret(T.GenResult(content="t</think>\nANSWER o0: yes\nANSWER o1: no\nANSWER o2: yes", finish_reason="stop"), q_options(row))
        self.assertEqual(p.answers, {"o0": "yes", "o1": "no", "o2": "yes"})


class RecordContract(unittest.TestCase):
    def setUp(self) -> None:
        self.row = prow(25, 0)
        self.prompt = P.build_prompt({"state": self.row["state"], "questions": self.row["questions"]})

    def record(self, gen, **kw):
        kw.setdefault("max_tokens", 1000)
        kw.setdefault("keep_reasoning", True)
        return T.make_record(self.row, self.prompt, gen, attempt=1, **kw)

    def test_ok_record(self) -> None:
        gen = T.GenResult(reasoning="think", content="ANSWER q: B", finish_reason="stop", prompt_tokens=50, completion_tokens=300, reasoning_tokens=280, seconds=12.5)
        r = self.record(gen)
        self.assertEqual((r["kind"], r["id"], r["benchmark"], r["status"], r["answers"]), ("row", "25:0", 25, "ok", {"q": "B"}))
        self.assertEqual((r["finish_reason"], r["truncated"], r["max_tokens"], r["attempt"]), ("stop", False, 1000, 1))
        self.assertEqual(r["tokens"], {"prompt": 50, "completion": 300, "reasoning": 280})
        self.assertEqual((r["seconds"], r["reasoning"], r["final"]), (12.5, "think", "ANSWER q: B"))

    def test_reasoning_can_be_dropped_and_the_final_is_capped(self) -> None:
        gen = T.GenResult(reasoning="think", content="x" * 5000 + "\nANSWER q: B", finish_reason="stop")
        r = self.record(gen, keep_reasoning=False)
        self.assertNotIn("reasoning", r)
        self.assertEqual(r["answers"], {"q": "B"})
        self.assertLessEqual(len(r["final"]), 2000)
        self.assertTrue(r["final"].endswith("ANSWER q: B"))

    def test_truncated_record(self) -> None:
        r = self.record(T.GenResult(content="hmm", finish_reason="length", completion_tokens=1000))
        self.assertEqual((r["status"], r["truncated"], r["answers"]), ("ok", True, {"q": None}))

    def test_whether_the_model_thought_is_measured_on_every_row(self) -> None:
        cases = [
            (T.GenResult(reasoning="long thinking", content="ANSWER q: B", finish_reason="stop"), True),              # the server split it off
            (T.GenResult(content="so B...\n</think>\nANSWER q: B", finish_reason="stop"), True),                      # inline, closed
            (T.GenResult(content="ANSWER q: B", finish_reason="stop", completion_tokens=6), False),                    # answered at once
            (T.GenResult(content="ANSWER q: B", finish_reason="stop", completion_tokens=99), False),                   # one token below the bar
            (T.GenResult(content="ANSWER q: B", finish_reason="stop", completion_tokens=C.MIN_THINKING_TOKENS), True),
            (T.GenResult(content="ANSWER q: B", finish_reason="stop", reasoning_tokens=300), True),                    # the server hides the text, not the count
            (T.GenResult(content="</think>\nANSWER q: B", finish_reason="stop", completion_tokens=8), False),          # a closing tag with nothing before it
            (T.GenResult(content="ANSWER q: B", finish_reason="stop"), False),                                         # no usage, no text: nothing shows thinking
            (T.GenResult(content="still thinking about it", finish_reason="length", completion_tokens=1000), True),
        ]
        for gen, expected in cases:
            self.assertIs(self.record(gen)["thought"], expected, gen)

    def test_a_stop_without_a_closing_tag_is_unreadable_and_recorded_as_such(self) -> None:
        r = self.record(T.GenResult(content="B probably.\nANSWER q: B", finish_reason="stop", completion_tokens=8))
        self.assertEqual((r["answers"], r["no_close"], r["thought"]), ({"q": None}, True, False))
        r = self.record(T.GenResult(content="x\n</think>\nANSWER q: B", finish_reason="stop"))
        self.assertEqual((r["answers"], r["no_close"]), ({"q": "B"}, False))
        r = self.record(T.GenResult(content="B probably.\nANSWER q: B", finish_reason="stop"), thinking=False)       # a control without thinking is read whole
        self.assertEqual((r["answers"], r["no_close"]), ({"q": "B"}, False))

    def test_how_each_answer_was_found_is_recorded(self) -> None:
        r = self.record(T.GenResult(content="t</think>\nANSWER q: B", finish_reason="stop"))
        self.assertEqual(r["read_by"], {"q": "marker"})
        r = self.record(T.GenResult(content="t</think>\nThe answer is C", finish_reason="stop"))
        self.assertEqual((r["answers"], r["read_by"]), ({"q": "C"}, {"q": "generic"}))
        r = self.record(T.GenResult(content="t</think>\nno idea", finish_reason="stop"))
        self.assertEqual((r["answers"], r["read_by"]), ({"q": None}, {"q": None}))

    def test_garbled_output_is_judged_when_the_row_is_written_so_it_survives_dropping_the_reasoning(self) -> None:
        r = self.record(T.GenResult(reasoning="!" * 3000, content="ANSWER q: B", finish_reason="stop"), keep_reasoning=False)
        self.assertNotIn("reasoning", r)
        self.assertIs(r["garbled"], True)
        r = self.record(T.GenResult(reasoning="the options differ in the second clause " * 20, content="ANSWER q: B", finish_reason="stop"), keep_reasoning=False)
        self.assertIs(r["garbled"], False)

    def test_the_endpoint_that_answered_is_on_the_row(self) -> None:
        gen = T.GenResult(content="t</think>\nANSWER q: B", finish_reason="stop")
        self.assertEqual(self.record(gen, endpoint="https://a.example/v1")["endpoint"], "https://a.example/v1")
        self.assertNotIn("endpoint", self.record(gen))
        self.assertEqual(self.record(T.GenResult(error="HTTP 503"), endpoint="https://a.example/v1")["endpoint"], "https://a.example/v1")

    def test_transient_error_is_status_error_and_a_row_error_is_failed(self) -> None:
        e = self.record(T.GenResult(error="HTTP 503", transient=True))
        f = self.record(T.GenResult(error="HTTP 400: context length", transient=False))
        self.assertEqual((e["status"], e["error"], e["answers"]), ("error", "HTTP 503", {"q": None}))
        self.assertEqual((f["status"], f["error"]), ("failed", "HTTP 400: context length"))
        self.assertTrue(all(k not in e for k in ("reasoning", "final")))


class ConfigContract(unittest.TestCase):
    def test_fingerprint_ignores_the_budget_the_seed_and_the_endpoint(self) -> None:
        a = T.fingerprint_of(fake_config())
        self.assertEqual(a, T.fingerprint_of(fake_config(max_tokens=99, seed=5, endpoint="http://other/v1")))

    def test_fingerprint_follows_what_changes_the_answers(self) -> None:
        a = T.fingerprint_of(fake_config())
        for over in ({"model": "other"}, {"revision": "abc"}, {"quantization": "nf4"}, {"dtype": "bfloat16"}, {"reasoning_effort": "low"}, {"backend": "x"},
                     {"prompt_version": "think-gt-prompt/2"}, {"chat_template_kwargs": {"enable_thinking": False}},
                     {"sampling": {**C.THINKING_SAMPLING, "temperature": 0.6}}, {"extra_body": {"a": 1}}):
            self.assertNotEqual(a, T.fingerprint_of(fake_config(**over)), over)

    def test_row_seed_is_stable_per_id(self) -> None:
        self.assertIsNone(T.row_seed(None, "25:0"))
        self.assertEqual(T.row_seed(7, "25:0"), T.row_seed(7, "25:0"))
        self.assertNotEqual(T.row_seed(7, "25:0"), T.row_seed(7, "25:1"))
        self.assertNotEqual(T.row_seed(7, "25:0"), T.row_seed(8, "25:0"))
        self.assertTrue(0 <= T.row_seed(7, "25:0") < 2 ** 31)


class StoreContract(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "results.jsonl"

    def open(self, **over):
        cfg = fake_config(**{k: v for k, v in over.items() if k not in ("pack", "overwrite")})
        return T.ResultStore.open(self.path, cfg, over.get("pack", "p" * 64), 10, overwrite=over.get("overwrite", False))

    @staticmethod
    def rec(i, status="ok", truncated=False, max_tokens=1000):
        return {"kind": "row", "id": f"25:{i}", "benchmark": 25, "status": status, "answers": {"q": "A"}, "truncated": truncated, "max_tokens": max_tokens}

    def test_new_file_starts_with_a_header(self) -> None:
        self.open()
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        h = json.loads(lines[0])
        self.assertEqual((h["kind"], h["format"], h["pack_content_sha256"]), ("header", C.RESULT_FORMAT, "p" * 64))
        self.assertEqual(h["config_fingerprint"], T.fingerprint_of(fake_config()))
        self.assertEqual(h["config"]["model"], "m")

    def test_reopening_with_the_same_config_resumes(self) -> None:
        s = self.open()
        s.append(self.rec(0))
        s.append(self.rec(1, status="error"))
        s.append(self.rec(2, status="failed"))
        s2 = self.open()
        self.assertEqual(s2.done_ids(1000), {"25:0", "25:2"})
        self.assertEqual(len(self.path.read_text(encoding="utf-8").splitlines()), 4)        # no second header

    def test_a_cut_last_line_is_repaired_and_appends_start_on_a_new_line(self) -> None:
        s = self.open()
        s.append(self.rec(0))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"kind":"row","id":"25:1","sta')
        s2 = self.open()
        s2.append(self.rec(1))
        lines = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([x.get("id") for x in lines], [None, "25:0", "25:1"])
        self.assertEqual(s2.done_ids(1000), {"25:0", "25:1"})

    def test_another_config_or_another_pack_is_refused_with_the_reason(self) -> None:
        self.open()
        with self.assertRaises(T.ResumeMismatch) as c:
            self.open(model="other")
        self.assertIn("model", str(c.exception))
        with self.assertRaises(T.ResumeMismatch) as c:
            self.open(pack="q" * 64)
        self.assertIn("pack", str(c.exception))

    def test_overwrite_starts_over(self) -> None:
        s = self.open()
        s.append(self.rec(0))
        s2 = self.open(model="other", overwrite=True)
        self.assertEqual(s2.done_ids(1000), set())
        self.assertEqual(len(self.path.read_text(encoding="utf-8").splitlines()), 1)

    def test_a_file_that_is_not_a_results_file_is_refused(self) -> None:
        self.path.write_text('{"hello": 1}\n', encoding="utf-8")
        with self.assertRaises(T.ResumeMismatch):
            self.open()

    def test_the_last_record_of_an_id_wins(self) -> None:
        s = self.open()
        s.append(self.rec(0, status="error"))
        s.append(self.rec(0, status="ok"))
        self.assertEqual(s.latest()["25:0"]["status"], "ok")
        self.assertEqual(s.done_ids(1000), {"25:0"})

    def test_only_the_errors_at_the_end_of_a_row_history_count_towards_giving_it_up(self) -> None:
        s = self.open()
        for status in ("error", "error"):
            s.append(self.rec(0, status=status))
        s.append(self.rec(1, status="error"))
        self.assertEqual(s.error_streaks(), {"25:0": 2, "25:1": 1})
        s.append(self.rec(0, status="ok"))                                  # a success starts the count over
        s.append(self.rec(0, status="error"))
        s.append(self.rec(1, status="failed"))                              # and so does a `failed` record
        s.append(self.rec(1, status="error"))
        self.assertEqual(s.error_streaks(), {"25:0": 1, "25:1": 1})
        s.append(self.rec(2, status="ok"))
        self.assertNotIn("25:2", s.error_streaks())

    def test_a_row_gets_three_tries_by_default(self) -> None:
        import inspect
        self.assertEqual(C.DEFAULT_MAX_ATTEMPTS, 3)
        self.assertEqual(T.build_parser().parse_args(["--pack", "p", "--out", "o"]).max_attempts, 3)
        self.assertEqual(inspect.signature(T.run).parameters["max_attempts"].default, 3)
        self.assertIn("3 erros seguidos", " ".join(readme_text().split()))

    def test_failed_rows_are_work_again_only_with_retry_failed(self) -> None:
        s = self.open()
        s.append(self.rec(0, status="failed"))
        s.append(self.rec(1, status="ok"))
        s.append(self.rec(2, status="error"))
        self.assertEqual(s.done_ids(1000), {"25:0", "25:1"})
        self.assertEqual(s.done_ids(1000, retry_failed=True), {"25:1"})

    def test_a_truncated_row_is_redone_only_under_a_bigger_budget(self) -> None:
        s = self.open()
        s.append(self.rec(0, truncated=True, max_tokens=100))
        s.append(self.rec(1, truncated=False, max_tokens=100))
        self.assertEqual(s.done_ids(100), {"25:0", "25:1"})
        self.assertEqual(s.done_ids(200), {"25:1"})

    def test_the_budget_cannot_go_down_on_resume(self) -> None:
        s = self.open(max_tokens=500)
        s.append(self.rec(0, max_tokens=500))
        self.assertEqual(self.open(max_tokens=800).max_tokens_seen(), 800)       # raising it is fine
        with self.assertRaises(T.ResumeMismatch) as c:
            self.open(max_tokens=300)
        self.assertIn("max_tokens", str(c.exception))


class FakeBackend:
    name = "fake"

    def __init__(self, reply=None, workers=1, batch_size=1, clock=None, step=0.0, sleep=0.0, raises=None):
        self.workers, self.batch_size = workers, batch_size
        self.reply = reply or (lambda item: T.GenResult(content="t</think>\nANSWER q: A", finish_reason="stop",
                                                         prompt_tokens=10, completion_tokens=20, seconds=1.0))
        self.clock, self.step, self.sleep, self.raises = clock, step, sleep, raises
        self.calls, self.inflight, self.max_inflight = [], 0, 0
        import threading
        self._lock = threading.Lock()

    def generate(self, items):
        with self._lock:
            self.calls.append([i.id for i in items])
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.sleep:
                import time
                time.sleep(self.sleep)
            if self.clock is not None:
                self.clock.t += self.step
            if self.raises is not None:
                raise self.raises
            return [self.reply(i) for i in items]
        finally:
            with self._lock:
                self.inflight -= 1


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class RunnerContract(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "results.jsonl"
        self.rows = [prow(25 if i % 2 == 0 else 58, i // 2, picks={"m": "A"}) for i in range(10)]
        self.logs = []

    def store(self):
        return T.ResultStore.open(self.path, fake_config(), "p" * 64, len(self.rows))

    def run_(self, backend, **kw):
        kw.setdefault("max_tokens", 1000)
        kw.setdefault("log", self.logs.append)
        return T.run(self.rows, backend, self.store(), **kw)

    def ids(self):
        return [r["id"] for r in T.ResultStore.read(self.path)[1]]

    def test_every_pending_row_is_run_in_pack_order_and_written(self) -> None:
        b = FakeBackend()
        st = self.run_(b)
        self.assertEqual(self.ids(), [r["id"] for r in self.rows])
        self.assertEqual((st.done, st.ok, st.errors, st.failed), (10, 10, 0, 0))
        self.assertEqual(b.calls, [[r["id"]] for r in self.rows])
        self.assertEqual(self.store().done_ids(1000), {r["id"] for r in self.rows})

    def test_done_rows_are_skipped_on_the_next_run(self) -> None:
        self.run_(FakeBackend(), limit=4)
        b = FakeBackend()
        st = self.run_(b)
        self.assertEqual(st.done, 6)
        self.assertEqual([c[0] for c in b.calls], [r["id"] for r in self.rows[4:]])
        self.assertEqual(self.ids(), [r["id"] for r in self.rows])            # no row twice
        self.assertEqual(self.run_(FakeBackend()).done, 0)

    def test_a_crash_between_rows_loses_nothing(self) -> None:
        class Boom(KeyboardInterrupt):
            pass
        calls = []

        def reply(item):
            calls.append(item.id)
            if len(calls) == 4:
                raise Boom()
            return T.GenResult(content="t</think>\nANSWER q: B", finish_reason="stop")
        with self.assertRaises(Boom):
            self.run_(FakeBackend(reply=reply))
        self.assertEqual(self.ids(), [r["id"] for r in self.rows[:3]])
        self.run_(FakeBackend())
        self.assertEqual(self.ids(), [r["id"] for r in self.rows])

    def test_limit_and_benchmark_filter(self) -> None:
        b = FakeBackend()
        self.run_(b, benchmarks=[58], limit=3)
        self.assertEqual([c[0] for c in b.calls], ["58:0", "58:1", "58:2"])

    def test_time_budget_stops_launching_new_rows(self) -> None:
        clock = Clock()
        b = FakeBackend(clock=clock, step=10.0)
        st = self.run_(b, time_budget_s=25, clock=clock)
        self.assertEqual(st.done, 3)                           # starts at 0, 10, 20; at 30 the budget is over
        self.assertEqual(st.stopped, "time budget")

    def test_batches_are_consecutive_rows_and_results_stay_aligned(self) -> None:
        def reply(item):
            return T.GenResult(content="t</think>\nANSWER q: " + ("B" if item.id.endswith("1") else "A"), finish_reason="stop")
        b = FakeBackend(batch_size=4, reply=reply)
        self.run_(b)
        self.assertEqual([len(c) for c in b.calls], [4, 4, 2])
        got = {r["id"]: r["answers"]["q"] for r in T.ResultStore.read(self.path)[1]}
        self.assertEqual(got["58:1"], "B")
        self.assertEqual(got["25:0"], "A")

    def test_concurrent_window(self) -> None:
        b = FakeBackend(workers=3, sleep=0.02)
        st = self.run_(b)
        self.assertEqual(st.done, 10)
        self.assertTrue(2 <= b.max_inflight <= 3, b.max_inflight)
        self.assertEqual(sorted(self.ids()), sorted(r["id"] for r in self.rows))

    def test_an_exception_in_the_backend_becomes_error_records_and_the_run_goes_on(self) -> None:
        n = {"k": 0}

        def reply(item):
            n["k"] += 1
            if n["k"] == 2:
                raise RuntimeError("cuda hiccup")
            return T.GenResult(content="t</think>\nANSWER q: A", finish_reason="stop")
        st = self.run_(FakeBackend(reply=reply))
        recs = T.ResultStore.read(self.path)[1]
        self.assertEqual((st.done, st.ok, st.errors), (10, 9, 1))
        bad = [r for r in recs if r["status"] == "error"]
        self.assertEqual(len(bad), 1)
        self.assertIn("cuda hiccup", bad[0]["error"])
        self.assertEqual(self.store().done_ids(1000), {r["id"] for r in self.rows} - {bad[0]["id"]})

    def test_the_run_aborts_after_too_many_consecutive_errors(self) -> None:
        st = self.run_(FakeBackend(reply=lambda item: T.GenResult(error="HTTP 503", transient=True)), max_consecutive_errors=3)
        self.assertEqual(st.done, 3)
        self.assertEqual(st.stopped, "3 consecutive errors")

    def test_a_backend_error_stops_the_run_at_once(self) -> None:
        with self.assertRaises(T.BackendError):
            self.run_(FakeBackend(raises=T.BackendError("HTTP 401: bad key")))
        self.assertEqual(self.ids(), [])

    # -- a row that cannot be answered is a wrong answer, not a row that is "absent" for ever (review of #78, blocker 1)
    def down(self):
        return FakeBackend(reply=lambda item: T.GenResult(error="HTTP 503: overloaded", transient=True))

    def test_a_row_that_errors_in_max_attempts_runs_is_given_up_as_failed(self) -> None:
        for k in (1, 2):
            st = self.run_(self.down(), max_attempts=3, limit=2, max_consecutive_errors=100)
            self.assertEqual((st.errors, st.failed), (2, 0), k)
            self.assertEqual(self.store().done_ids(1000), set(), "a row with an error is still pending")
        st = self.run_(self.down(), max_attempts=3, limit=2, max_consecutive_errors=100)
        self.assertEqual((st.errors, st.failed), (0, 2))
        last = self.store().latest()
        for rid in ("25:0", "58:0"):
            self.assertEqual(last[rid]["status"], "failed")
            self.assertEqual(last[rid]["gave_up"], 3)
            self.assertIn("HTTP 503", last[rid]["error"])
            self.assertIn("3 attempts", last[rid]["error"])
        self.assertEqual(self.store().done_ids(1000), {"25:0", "58:0"})
        self.assertEqual(self.run_(self.down(), max_attempts=3, limit=2).done, 2, "the next two rows are tried, not the given-up ones")

    def test_one_attempt_makes_the_first_error_final_and_a_success_in_between_keeps_the_row_alive(self) -> None:
        self.run_(self.down(), max_attempts=1, limit=1)
        self.assertEqual(self.store().latest()["25:0"]["status"], "failed")
        self.run_(self.down(), max_attempts=3, limit=1, benchmarks=[58])
        self.run_(FakeBackend(), max_attempts=3, limit=1, benchmarks=[58])
        self.run_(self.down(), max_attempts=3, limit=1, benchmarks=[58])                  # already ok: not run again
        self.assertEqual(self.store().latest()["58:0"]["status"], "ok")

    def test_only_the_errors_at_the_end_of_the_history_count(self) -> None:
        # a row cut by the budget (ok, truncated) and redone under a bigger one starts its count over
        self.run_(FakeBackend(reply=lambda item: T.GenResult(content="hmm", finish_reason="length")), limit=1, max_tokens=100)
        for k in range(2):
            self.run_(self.down(), max_attempts=3, limit=1, max_tokens=200, max_consecutive_errors=100)
        self.assertEqual(self.store().latest()["25:0"]["status"], "error")
        self.run_(self.down(), max_attempts=3, limit=1, max_tokens=200, max_consecutive_errors=100)
        self.assertEqual(self.store().latest()["25:0"]["status"], "failed")

    def test_retry_failed_gives_the_given_up_rows_a_fresh_count(self) -> None:
        self.run_(self.down(), max_attempts=1, limit=1)
        self.assertEqual(self.run_(FakeBackend(), limit=1).done, 1, "the next row, not the given-up one")
        st = self.run_(self.down(), max_attempts=2, retry_failed=True, limit=1)
        self.assertEqual((st.errors, st.failed), (1, 0), "the count starts over, so one error is not final yet")
        self.run_(FakeBackend(), retry_failed=True, limit=1)
        self.assertEqual(self.store().latest()["25:0"]["status"], "ok")

    def test_a_given_up_row_still_counts_toward_the_consecutive_errors_that_stop_the_run(self) -> None:
        st = self.run_(self.down(), max_attempts=1, max_consecutive_errors=3)
        self.assertEqual((st.done, st.failed, st.stopped), (3, 3, "3 consecutive errors"))

    def test_the_progress_line_flags_a_row_that_did_not_think_and_one_that_did_not_close(self) -> None:
        def reply(item):
            return T.GenResult(content="ANSWER q: A", finish_reason="stop", completion_tokens=5)
        self.run_(FakeBackend(reply=reply), limit=1)
        line = self.logs[-1]
        self.assertIn("no_thinking", line)
        self.assertIn("no_close", line)
        self.logs.clear()
        self.run_(FakeBackend(), limit=1, benchmarks=[58])
        self.assertNotIn("no_thinking", self.logs[-1])
        self.assertNotIn("no_close", self.logs[-1])

    def test_rows_are_written_with_the_thinking_the_config_asked_for_and_the_endpoint_of_the_config(self) -> None:
        def bare(item):
            return T.GenResult(content="B...\nANSWER q: B", finish_reason="stop", completion_tokens=9)
        self.run_(FakeBackend(reply=bare), limit=1)
        rec = self.store().latest()["25:0"]
        self.assertEqual((rec["answers"], rec["no_close"], rec["thought"]), ({"q": None}, True, False))
        off = fake_config(chat_template_kwargs={"enable_thinking": False}, endpoint="https://a.example/v1")
        path = Path(self.tmp.name) / "control.jsonl"
        T.run(self.rows, FakeBackend(reply=bare), T.ResultStore.open(path, off, "p" * 64, 10), max_tokens=1000, limit=1, log=lambda s: None)
        rec = T.ResultStore.read(path)[1][0]
        self.assertEqual((rec["answers"], rec["no_close"], rec["endpoint"]), ({"q": "B"}, False, "https://a.example/v1"))

    def test_the_thinking_text_is_not_stored_unless_asked(self) -> None:
        self.run_(FakeBackend(reply=lambda item: T.GenResult(reasoning="secret reasoning", content="ANSWER q: A", finish_reason="stop")), limit=1)
        self.assertNotIn("reasoning", self.store().latest()["25:0"])
        path = Path(self.tmp.name) / "kept.jsonl"
        T.run(self.rows, FakeBackend(reply=lambda item: T.GenResult(reasoning="secret reasoning", content="ANSWER q: A", finish_reason="stop")),
              T.ResultStore.open(path, fake_config(), "p" * 64, 10), max_tokens=1000, limit=1, keep_reasoning=True, log=lambda s: None)
        self.assertEqual(T.ResultStore.read(path)[1][0]["reasoning"], "secret reasoning")

    def test_stop_flag(self) -> None:
        flag = {"stop": False}

        def reply(item):
            flag["stop"] = item.id == "58:0"
            return T.GenResult(content="t</think>\nANSWER q: A", finish_reason="stop")
        st = self.run_(FakeBackend(reply=reply), stop=lambda: flag["stop"])
        self.assertEqual(st.done, 2)
        self.assertEqual(st.stopped, "stop requested")

    def test_stats_count_unparseable_truncated_and_tokens(self) -> None:
        def reply(item):
            i = int(item.id.split(":")[1])
            if item.id.startswith("25") and i == 0:
                return T.GenResult(content="I do not know</think>\nno idea", finish_reason="stop", completion_tokens=5)
            if item.id.startswith("25") and i == 1:
                return T.GenResult(content="hmm hmm", finish_reason="length", completion_tokens=1000)
            return T.GenResult(content="t</think>\nANSWER q: A", finish_reason="stop", completion_tokens=7)
        st = self.run_(FakeBackend(reply=reply))
        self.assertEqual((st.unparseable, st.truncated, st.completion_tokens), (2, 1, 5 + 1000 + 8 * 7))

    def test_progress_log_has_counts_and_never_the_question(self) -> None:
        self.run_(FakeBackend(), limit=2)
        text = "\n".join(self.logs)
        self.assertIn("1/2", text)
        self.assertIn("2/2", text)
        self.assertNotIn("text of", text)

    def test_prompt_given_to_the_backend_is_the_built_prompt(self) -> None:
        seen = []

        def reply(item):
            seen.append(item)
            return T.GenResult(content="t</think>\nANSWER q: A", finish_reason="stop")
        self.run_(FakeBackend(reply=reply), limit=1, seed=9)
        row = self.rows[0]
        self.assertEqual(seen[0].prompt, P.build_prompt({"state": row["state"], "questions": row["questions"]}).text)
        self.assertEqual(seen[0].seed, T.row_seed(9, row["id"]))


class FakeResponse:
    def __init__(self, body=b"", lines=None, status=200):
        self.body, self.lines, self.status = body, lines or [], status

    def read(self):
        return self.body

    def __iter__(self):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body="", headers=None):
    import io
    import urllib.error
    return urllib.error.HTTPError("http://x/v1/chat/completions", code, "err", headers or {}, io.BytesIO(body.encode()))


class OpenAIPayloadContract(unittest.TestCase):
    def backend(self, **kw):
        kw.setdefault("base_url", "http://127.0.0.1:8000/v1")
        kw.setdefault("model", "Qwen/Qwen3.8-27B")
        kw.setdefault("sleep", lambda s: None)
        return T.OpenAIBackend(**kw)

    def test_default_payload_is_the_model_card_thinking_request(self) -> None:
        body = self.backend(max_tokens=777).payload(T.Item("25:0", "the prompt"))
        self.assertEqual(body["model"], "Qwen/Qwen3.8-27B")
        self.assertEqual(body["messages"], [{"role": "user", "content": "the prompt"}])
        self.assertEqual(body["max_tokens"], 777)
        for k, v in C.THINKING_SAMPLING.items():
            self.assertEqual(body[k], v, k)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertNotIn("reasoning_effort", body)
        self.assertNotIn("seed", body)

    def test_effort_seed_and_extra_body(self) -> None:
        body = self.backend(reasoning_effort="medium", extra_body={"foo": 1}).payload(T.Item("a", "p", seed=42))
        self.assertEqual((body["reasoning_effort"], body["seed"], body["top_k"], body["foo"]), ("medium", 42, 20, 1))

    def test_standard_only_drops_what_strict_servers_reject(self) -> None:
        body = self.backend(standard_only=True).payload(T.Item("a", "p"))
        for k in ("top_k", "min_p", "repetition_penalty", "chat_template_kwargs"):
            self.assertNotIn(k, body)
        self.assertEqual((body["temperature"], body["top_p"]), (1.0, 0.95))

    def test_no_stream(self) -> None:
        body = self.backend(stream=False).payload(T.Item("a", "p"))
        self.assertNotIn("stream", body)
        self.assertNotIn("stream_options", body)

    def test_url_and_auth_header(self) -> None:
        b = self.backend(base_url="https://h:1/v1/", api_key="sk-secret")
        self.assertEqual(b.url, "https://h:1/v1/chat/completions")
        self.assertEqual(b.headers()["Authorization"], "Bearer sk-secret")
        self.assertNotIn("Authorization", self.backend().headers())
        self.assertNotIn("sk-secret", repr(b.describe()))

    def test_http_error_classes(self) -> None:
        for code in (408, 425, 429, 500, 502, 503, 504):
            self.assertEqual(T.classify_http_error(code, ""), "retry", code)
        for code in (400, 413, 422):
            self.assertEqual(T.classify_http_error(code, "This model's maximum context length is 40960 tokens"), "row", code)
        self.assertEqual(T.classify_http_error(400, "Unknown parameter: top_k"), "fatal")
        for code in (401, 403, 404):
            self.assertEqual(T.classify_http_error(code, "nope"), "fatal", code)


class OpenAIRequestContract(unittest.TestCase):
    OK = json.dumps({"choices": [{"message": {"role": "assistant", "content": "t</think>\nANSWER q: B", "reasoning_content": None},
                                  "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 22}}).encode()

    def backend(self, opener, **kw):
        self.sleeps = []
        return T.OpenAIBackend(base_url="http://h/v1", model="m", opener=opener, sleep=self.sleeps.append, stream=kw.pop("stream", False), **kw)

    def test_plain_json_reply(self) -> None:
        g = self.backend(lambda req, timeout: FakeResponse(self.OK)).generate([T.Item("a", "p")])[0]
        self.assertEqual((g.content, g.finish_reason, g.prompt_tokens, g.completion_tokens, g.error), ("t</think>\nANSWER q: B", "stop", 11, 22, None))

    def test_reasoning_field_of_vllm_and_its_older_name(self) -> None:
        for name in ("reasoning", "reasoning_content"):
            body = json.dumps({"choices": [{"message": {"content": "ANSWER q: B", name: "thinking"}, "finish_reason": "stop"}],
                               "usage": {"completion_tokens_details": {"reasoning_tokens": 9}, "completion_tokens": 20}}).encode()
            g = self.backend(lambda req, timeout, body=body: FakeResponse(body)).generate([T.Item("a", "p")])[0]
            self.assertEqual((g.reasoning, g.content, g.reasoning_tokens, g.completion_tokens), ("thinking", "ANSWER q: B", 9, 20), name)

    def test_stream_reply_joins_reasoning_and_content_and_reads_usage(self) -> None:
        def sse(obj):
            return b"data: " + json.dumps(obj).encode() + b"\n"
        lines = [b": keep-alive\n", sse({"choices": [{"delta": {"role": "assistant"}}]}),
                 sse({"choices": [{"delta": {"reasoning": "think "}}]}), sse({"choices": [{"delta": {"reasoning_content": "more"}}]}),
                 b"\n", sse({"choices": [{"delta": {"content": "ANSWER q: "}}]}), sse({"choices": [{"delta": {"content": "C"}, "finish_reason": "stop"}]}),
                 sse({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 8}}), b"data: [DONE]\n"]
        g = self.backend(lambda req, timeout: FakeResponse(lines=lines), stream=True).generate([T.Item("a", "p")])[0]
        self.assertEqual((g.reasoning, g.content, g.finish_reason, g.prompt_tokens, g.completion_tokens), ("think more", "ANSWER q: C", "stop", 5, 8))

    def test_a_stream_that_ends_without_finish_is_retried(self) -> None:
        cut = [b'data: {"choices": [{"delta": {"content": "AN"}}]}\n']
        full = [b'data: {"choices": [{"delta": {"content": "ANSWER q: A"}, "finish_reason": "stop"}]}\n', b"data: [DONE]\n"]
        replies = iter([FakeResponse(lines=cut), FakeResponse(lines=full)])
        g = self.backend(lambda req, timeout: next(replies), stream=True).generate([T.Item("a", "p")])[0]
        self.assertEqual((g.content, g.error), ("ANSWER q: A", None))

    def test_transient_http_errors_are_retried_with_backoff_then_succeed(self) -> None:
        replies = iter([http_error(503), http_error(429, headers={"Retry-After": "7"}), FakeResponse(self.OK)])

        def opener(req, timeout):
            r = next(replies)
            if isinstance(r, Exception):
                raise r
            return r
        b = self.backend(opener, retries=4)
        g = b.generate([T.Item("a", "p")])[0]
        self.assertIsNone(g.error)
        self.assertEqual(len(self.sleeps), 2)
        self.assertEqual(self.sleeps[1], 7.0)                  # Retry-After is honored
        self.assertGreater(self.sleeps[0], 0)

    def test_retries_run_out_into_a_transient_error(self) -> None:
        def opener(req, timeout):
            raise http_error(500, "boom")
        g = self.backend(opener, retries=2).generate([T.Item("a", "p")])[0]
        self.assertTrue(g.transient)
        self.assertIn("500", g.error)
        self.assertEqual(len(self.sleeps), 2)

    def test_network_errors_are_transient(self) -> None:
        import urllib.error

        def opener(req, timeout):
            raise urllib.error.URLError("connection refused")
        g = self.backend(opener, retries=1).generate([T.Item("a", "p")])[0]
        self.assertTrue(g.transient)
        self.assertIn("connection refused", g.error)

    def test_a_context_overflow_fails_the_row_not_the_run(self) -> None:
        def opener(req, timeout):
            raise http_error(400, '{"error": {"message": "maximum context length is 4096 tokens"}}')
        g = self.backend(opener).generate([T.Item("a", "p")])[0]
        self.assertFalse(g.transient)
        self.assertIn("context length", g.error)

    def test_bad_key_or_bad_parameter_stops_the_run(self) -> None:
        for code, msg in ((401, "bad key"), (404, "model not found"), (400, "Unknown parameter: top_k")):
            def opener(req, timeout, code=code, msg=msg):
                raise http_error(code, msg)
            with self.assertRaises(T.BackendError) as c:
                self.backend(opener).generate([T.Item("a", "p")])
            self.assertIn(str(code), str(c.exception))


class GateCurveContract(unittest.TestCase):
    @staticmethod
    def scenario_with_confidence():
        # 20 rows.  The decider is sure (0.9) and right on 10 easy rows; it is unsure (0.4) on 10 hard rows and right on 3 of them.
        # The thinker is right on all 10 hard rows and on 8 of the easy ones.
        rows, recs = [], []
        for i in range(20):
            easy = i < 10
            decider_right = easy or i < 13
            row = prow(25, i, picks={"m": "A" if decider_right else "B"})
            conf = 0.9 if easy else 0.4
            keys = list(row["questions"]["q"]["criteria"])
            pick = "A" if decider_right else "B"
            row["decisions"]["m"]["q"]["probs"] = {k: (conf if k == pick else (1 - conf) / 3) for k in keys}
            thinker_right = (not easy) or i < 8
            rows.append(row)
            recs.append(trec(row, "A" if thinker_right else "B"))
        return rows, recs

    def test_confidence_of_a_row_is_its_weakest_question(self) -> None:
        row = {"questions": {"a": {}, "b": {}}, "decisions": {"m": {"a": {"probs": {"x": .9, "y": .1}}, "b": {"probs": {"x": .6, "y": .4}}}}}
        self.assertAlmostEqual(K.row_confidence(row, "m"), 0.6)

    def test_curve_runs_from_the_decider_alone_to_the_thinker_alone(self) -> None:
        s = K.score_pack(*self.scenario_with_confidence())
        pts = {p["tau"]: p for p in K.gate_curve(s, "m", thresholds=(0.0, 0.5, 1.01))}
        self.assertEqual((pts[0.0]["routed"], pts[1.01]["routed"]), (0.0, 1.0))
        self.assertAlmostEqual(pts[0.0]["macro_acc"], s.macro_acc("m"))
        self.assertAlmostEqual(pts[1.01]["macro_acc"], s.macro_acc(K.THINKER))

    def test_a_gate_on_confidence_beats_both_when_the_thinker_wins_where_the_decider_is_unsure(self) -> None:
        s = K.score_pack(*self.scenario_with_confidence())
        mid = next(p for p in K.gate_curve(s, "m", thresholds=(0.5,)))
        self.assertAlmostEqual(mid["routed"], 0.5)                       # only the 10 unsure rows go to the thinker
        self.assertAlmostEqual(mid["macro_acc"], (10 + 10) / 20)        # easy rows from the decider (10 right) + hard rows from the thinker (10 right)
        self.assertGreater(mid["macro_acc"], s.macro_acc("m"))           # decider alone: 13 of 20
        self.assertGreater(mid["macro_acc"], s.macro_acc(K.THINKER))     # thinker alone: 18 of 20 -> 0.9 vs hybrid 1.0 here
        self.assertAlmostEqual(mid["delta_vs_reference_pts"], 100 * (mid["macro_acc"] - s.macro_acc("m")))

    def test_threshold_is_strict_less_than(self) -> None:
        s = K.score_pack(*self.scenario_with_confidence())
        pts = K.gate_curve(s, "m", thresholds=(0.4, 0.9))
        self.assertEqual(pts[0]["routed"], 0.0)                          # confidence 0.4 is not < 0.4
        self.assertAlmostEqual(pts[1]["routed"], 0.5)                    # 0.9 is not < 0.9, 0.4 is

    def test_report_carries_the_curve_for_the_reference(self) -> None:
        rep = K.build_report(*self.scenario_with_confidence(), B=50, seed=1)
        self.assertEqual(rep["gate_curve"][0]["tau"], K.GATE_THRESHOLDS[0])
        self.assertEqual(len(rep["gate_curve"]), len(K.GATE_THRESHOLDS))
        md = K.render_markdown(rep)
        self.assertIn("Gate offline", md)


class EvalOnlyNoticeContract(unittest.TestCase):
    """ADR 0004, item 6: the GT sample and the thinker's traces on it are evaluation data, never training data."""

    def test_the_notice_names_the_rule(self) -> None:
        text = C.EVAL_ONLY_NOTICE.lower()
        for word in ("evaluation", "never", "training", "adr 0004"):
            self.assertIn(word, text)

    def test_results_header_carries_it(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            T.ResultStore.open(path, fake_config(), "p" * 64, 10)
            header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(header["notice"], C.EVAL_ONLY_NOTICE)

    def test_the_notice_does_not_change_the_config_fingerprint(self) -> None:
        self.assertEqual(T.fingerprint_of(fake_config()), T.fingerprint_of(fake_config()))
        self.assertNotIn("notice", fake_config())

    def test_report_markdown_says_it_in_portuguese(self) -> None:
        rows, recs = scenario({25: [(True, {"m": True})] * 4})
        md = K.render_markdown(K.build_report(rows, recs, B=20, seed=1))
        self.assertIn("nunca entram em treino", md)
        self.assertIn("ADR 0004", md)


class LatencySlackContract(unittest.TestCase):
    """The kit refuses a model whose mean latency passes 1000 ms; thinking is slow, so only a few requests may think."""

    @staticmethod
    def stats(seconds):
        return {"rows": len(seconds), "seconds": K._summary(seconds)}

    def test_summary_has_the_mean(self) -> None:
        self.assertAlmostEqual(K._summary([10.0, 20.0, 60.0])["mean"], 30.0)
        self.assertIsNone(K._summary([])["mean"])

    def test_fraction_is_the_rule_over_the_mean_time_of_a_thinking_request(self) -> None:
        got = K.latency_slack(self.stats([10.0, 20.0, 30.0]))                 # mean 20 s
        self.assertAlmostEqual(got["mean_seconds"], 20.0)
        self.assertAlmostEqual(got["max_thinking_fraction"], 1000.0 / 20000.0)
        self.assertEqual(got["rule_ms"], 1000.0)
        self.assertEqual(got["unit"], "mean latency per request, decider time ignored")

    def test_the_rule_is_read_on_the_mean_not_the_median(self) -> None:
        got = K.latency_slack(self.stats([1.0, 1.0, 100.0]))                # median 1 s, mean 34 s: the slow tail is what breaks the mean
        self.assertAlmostEqual(got["mean_seconds"], 34.0)
        self.assertAlmostEqual(got["max_thinking_fraction"], 1000.0 / 34000.0)

    def test_it_never_passes_one(self) -> None:
        self.assertEqual(K.latency_slack(self.stats([0.2, 0.4]))["max_thinking_fraction"], 1.0)

    def test_no_timing_no_answer(self) -> None:
        self.assertIsNone(K.latency_slack(self.stats([])))
        self.assertIsNone(K.latency_slack({"rows": 0, "seconds": K._summary([])}))

    def test_a_zero_mean_is_not_a_division_by_zero(self) -> None:
        self.assertEqual(K.latency_slack(self.stats([0.0, 0.0]))["max_thinking_fraction"], 1.0)

    def test_the_rule_is_a_parameter(self) -> None:
        got = K.latency_slack(self.stats([10.0]), rule_ms=500.0)
        self.assertAlmostEqual(got["max_thinking_fraction"], 0.05)

    @staticmethod
    def timed_scenario(seconds):
        rows, recs = [], []
        for i in range(20):
            easy = i < 10
            row = prow(25, i, picks={"m": "A" if easy or i < 13 else "B"})
            conf = 0.9 if easy else 0.4
            keys = list(row["questions"]["q"]["criteria"])
            pick = "A" if easy or i < 13 else "B"
            row["decisions"]["m"]["q"]["probs"] = {k: (conf if k == pick else (1 - conf) / 3) for k in keys}
            rows.append(row)
            recs.append(trec(row, "A", seconds=seconds))
        return rows, recs

    def test_report_marks_the_gate_points_that_fit_the_rule(self) -> None:
        rep = K.build_report(*self.timed_scenario(20.0), B=50, seed=1)
        self.assertAlmostEqual(rep["latency"]["max_thinking_fraction"], 0.05)
        fits = {g["tau"]: g["within_latency_rule"] for g in rep["gate_curve"]}
        self.assertTrue(fits[0.0])                               # nothing goes to the thinker
        self.assertFalse(fits[0.5])                              # half of the rows would
        self.assertFalse(fits[1.01])

    def test_a_fast_backend_fits_every_point(self) -> None:
        rep = K.build_report(*self.timed_scenario(0.5), B=50, seed=1)
        self.assertTrue(all(g["within_latency_rule"] for g in rep["gate_curve"]))

    def test_markdown_says_the_numbers_are_not_the_boards(self) -> None:
        md = K.render_markdown(K.build_report(*self.timed_scenario(20.0), B=50, seed=1))
        self.assertIn("1.000 ms", md)
        self.assertIn("UNVERIFIED", md)
        self.assertIn("5.0%", md)

    def test_without_timings_the_report_has_no_latency_block(self) -> None:
        rows, recs = self.timed_scenario(20.0)
        for r in recs:
            r.pop("seconds")
        rep = K.build_report(rows, recs, B=50, seed=1)
        self.assertIsNone(rep["latency"])
        self.assertTrue(all("within_latency_rule" not in g or g["within_latency_rule"] is None for g in rep["gate_curve"]))
        self.assertNotIn("Folga de latencia", K.render_markdown(rep))


class ThinkerConfigInReportContract(unittest.TestCase):
    """The report says which thinker produced the numbers, and warns where the run is not the plain gate."""

    @staticmethod
    def md(**over):
        rows, recs = scenario({25: [(True, {"m": False})] * 6})
        cfg = fake_config(model=C.QWEN_MODEL, revision=C.QWEN_REVISION, backend="openai", quantization=None)
        cfg.update(over)
        rep = K.build_report(rows, recs, header={"config": cfg, "config_fingerprint": "f"}, B=20, seed=1)
        return K.render_markdown(rep)

    def test_the_line_names_model_revision_backend_and_budget(self) -> None:
        md = self.md(max_tokens=16384)
        self.assertIn(f"Thinker: {C.QWEN_MODEL}@{C.QWEN_REVISION[:7]}", md)
        self.assertIn("backend openai", md)
        self.assertIn("max_tokens 16384", md)
        self.assertIn("thinking ligado", md)

    def test_a_pinned_unquantized_thinker_gets_no_warning(self) -> None:
        md = self.md(backend="openai", quantization="none")
        self.assertNotIn("AVISO: o thinker rodou quantizado", md)
        self.assertNotIn("AVISO: thinking desligado", md)
        self.assertNotIn("AVISO: modelo ou revisao", md)

    def test_a_quantized_thinker_is_a_one_sided_test(self) -> None:
        md = self.md(backend="transformers", quantization="nf4-fp16")
        self.assertIn("AVISO: o thinker rodou quantizado (nf4-fp16)", md)
        self.assertIn("um PASS vale", md)
        self.assertIn("FAIL nao prova", md)

    def test_thinking_off_is_flagged_as_not_the_gate(self) -> None:
        md = self.md(chat_template_kwargs={"enable_thinking": False})
        self.assertIn("thinking desligado", md)
        self.assertIn("AVISO: thinking desligado", md)

    def test_another_model_or_revision_is_flagged(self) -> None:
        self.assertIn("AVISO: modelo ou revisao", self.md(model="someone/else"))
        self.assertIn("AVISO: modelo ou revisao", self.md(revision="0" * 40))
        self.assertIn("AVISO: modelo ou revisao", self.md(revision=None))

    def test_a_report_without_a_header_still_renders(self) -> None:
        rows, recs = scenario({25: [(True, {"m": False})] * 6})
        md = K.render_markdown(K.build_report(rows, recs, B=20, seed=1))
        self.assertIn("Thinker: configuracao desconhecida", md)


class DigestContract(unittest.TestCase):
    """The manifest names its inputs by the bytes as stored (comparable to the Hub's LFS id) and by content (comparable to the kit's pins)."""

    def test_plain_file_has_the_stored_digest_only(self) -> None:
        import hashlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.jsonl"
            p.write_bytes(b'{"a":1}\n')
            self.assertEqual(C.file_digests(p), {"sha256": hashlib.sha256(b'{"a":1}\n').hexdigest()})

    def test_gz_file_adds_the_content_digest_that_survives_recompression(self) -> None:
        import gzip
        import hashlib
        import tempfile
        data = b'{"a":1}\n' * 500
        with tempfile.TemporaryDirectory() as tmp:
            fast, best = Path(tmp) / "fast.jsonl.gz", Path(tmp) / "best.jsonl.gz"
            for path, level in ((fast, 1), (best, 9)):
                with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=level, mtime=0) as g:
                    g.write(data)
            a, b = C.file_digests(fast), C.file_digests(best)
            self.assertEqual(a["uncompressed_sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(a["uncompressed_sha256"], b["uncompressed_sha256"])
            self.assertEqual(a["sha256"], hashlib.sha256(fast.read_bytes()).hexdigest())
            self.assertNotEqual(a["sha256"], b["sha256"])


class TraceContract(unittest.TestCase):
    """The `Raciocínio:` block that the hybrid appends to the state (ADR 0004, item 6; format in SCHEMA.md section 10, issue #69)."""

    def test_exact_layout(self) -> None:
        self.assertEqual(TR.MARKER, "Raciocínio:")
        self.assertEqual(TR.attach_trace("state text", "step 1\nstep 2"), "state text\n\nRaciocínio:\nstep 1\nstep 2")

    def test_the_state_loses_only_trailing_whitespace(self) -> None:
        self.assertEqual(TR.attach_trace("a\n  b \n\n", "t"), "a\n  b\n\nRaciocínio:\nt")

    def test_an_empty_state_gets_no_leading_blank_lines(self) -> None:
        self.assertEqual(TR.attach_trace("", "t"), "Raciocínio:\nt")
        self.assertEqual(TR.attach_trace("  \n", "t"), "Raciocínio:\nt")

    def test_an_empty_trace_adds_no_block(self) -> None:
        for empty in ("", "   ", "\n\n", None):
            self.assertEqual(TR.attach_trace("state", empty), "state")

    def test_a_state_that_already_has_a_block_is_refused(self) -> None:
        once = TR.attach_trace("state", "t")
        with self.assertRaises(TR.TraceError):
            TR.attach_trace(once, "other")

    def test_split_is_the_inverse_of_attach(self) -> None:
        for state in ("state", "multi\nline\n\nstate", "", "tem acentos: ação, coração"):
            for trace in ("t", "a\n\nb", "linha 1\n  recuo\nlinha 3"):
                self.assertEqual(TR.split_trace(TR.attach_trace(state, trace)), (state.rstrip(), trace))

    def test_split_without_a_block(self) -> None:
        self.assertEqual(TR.split_trace("just a state"), ("just a state", None))
        self.assertEqual(TR.split_trace(""), ("", None))
        self.assertFalse(TR.has_trace("just a state"))

    def test_the_last_marker_line_starts_the_block(self) -> None:
        self.assertEqual(TR.split_trace("a\nRaciocínio:\nx\nRaciocínio:\ny"), ("a\nRaciocínio:\nx", "y"))
        self.assertTrue(TR.has_trace("a\nRaciocínio:\nx\nRaciocínio:\ny"))

    def test_the_marker_must_be_a_whole_line(self) -> None:
        self.assertEqual(TR.split_trace("o Raciocínio: foi bom")[1], None)
        self.assertEqual(TR.split_trace("x\nRaciocínio: inline")[1], None)
        self.assertFalse(TR.has_trace("Raciocínio:x"))

    def test_the_marker_is_found_in_decomposed_unicode_too(self) -> None:
        import unicodedata
        nfd = unicodedata.normalize("NFD", "Raciocínio:")
        self.assertNotEqual(nfd, TR.MARKER)
        self.assertEqual(TR.split_trace(f"s\n\n{nfd}\nt"), ("s", "t"))
        self.assertTrue(TR.has_trace(f"s\n{nfd}\nt"))

    def test_a_marker_line_inside_the_trace_is_neutralized(self) -> None:
        trace = TR.render_trace("antes\nRaciocínio:\ndepois")
        self.assertFalse(TR.has_trace(trace))
        self.assertEqual(TR.split_trace(TR.attach_trace("s", trace)), ("s", trace))
        self.assertIn("Raciocínio :", trace)

    def test_text_is_normalized(self) -> None:
        got = TR.render_trace("a  \r\nb\r\n\r\n\r\n\r\nc\n")
        self.assertEqual(got, "a\nb\n\nc")

    def test_modes(self) -> None:
        self.assertEqual(TR.render_trace("R", "F", mode="reasoning"), "R")
        self.assertEqual(TR.render_trace("R", "F", mode="final"), "F")
        self.assertEqual(TR.render_trace("R", "F", mode="both"), "R\n\nF")
        self.assertEqual(TR.render_trace(None, "F", mode="both"), "F")
        self.assertEqual(TR.render_trace("R", None, mode="final"), "")
        with self.assertRaises(ValueError):
            TR.render_trace("R", "F", mode="all")

    def test_a_short_trace_is_untouched_and_a_long_one_keeps_its_tail(self) -> None:
        lines = [f"line {i:03d}" for i in range(100)]
        text = "\n".join(lines)
        self.assertEqual(TR.render_trace(text, max_chars=len(text)), text)
        clipped = TR.render_trace(text, max_chars=200)
        self.assertLessEqual(len(clipped), 200)
        self.assertTrue(clipped.endswith("line 099"))
        self.assertNotIn("line 000", clipped)
        first = clipped.splitlines()[0]
        self.assertTrue(first.startswith("[...") and "omitidos" in first, first)
        self.assertNotIn(first, text)
        for line in clipped.splitlines()[1:]:
            self.assertIn(line, lines)                    # the cut is on a line boundary

    def test_the_clip_reports_how_much_was_dropped(self) -> None:
        text = "x" * 1000
        clipped = TR.render_trace(text, max_chars=300)
        self.assertLessEqual(len(clipped), 300)
        dropped = int(re.search(r"(\d+) caracteres", clipped.splitlines()[0])[1])
        self.assertEqual(dropped + len(clipped.split("\n", 1)[1]), 1000)

    def test_one_endless_line_is_still_clipped(self) -> None:
        clipped = TR.render_trace("y" * 5000, max_chars=100)
        self.assertLessEqual(len(clipped), 100)
        self.assertTrue(clipped.endswith("y"))

    def test_a_budget_too_small_for_the_notice_still_obeys_it(self) -> None:
        self.assertLessEqual(len(TR.render_trace("z" * 500, max_chars=10)), 10)
        self.assertEqual(TR.render_trace("abc", max_chars=0), "")

    def test_default_budget(self) -> None:
        self.assertEqual(TR.DEFAULT_MAX_CHARS, 8000)
        self.assertLessEqual(len(TR.render_trace("w\n" * 20000)), 8000)

    def test_rendering_is_deterministic(self) -> None:
        text = "\n".join(f"s{i}" for i in range(500))
        self.assertEqual(TR.render_trace(text, max_chars=400), TR.render_trace(text, max_chars=400))

    def test_from_record(self) -> None:
        ok = {"status": "ok", "truncated": False, "reasoning": "think", "final": "answer"}
        self.assertEqual(TR.from_record(ok), "think")
        self.assertEqual(TR.from_record(ok, mode="final"), "answer")
        self.assertEqual(TR.from_record(ok, mode="both"), "think\n\nanswer")
        self.assertIsNone(TR.from_record({**ok, "status": "error"}))
        self.assertIsNone(TR.from_record({**ok, "status": "failed"}))
        self.assertIsNone(TR.from_record({**ok, "truncated": True}))                  # the conclusion is missing
        self.assertEqual(TR.from_record({**ok, "truncated": True}, allow_truncated=True), "think")
        self.assertIsNone(TR.from_record({"status": "ok", "final": "answer"}))        # run with --no-reasoning
        self.assertEqual(TR.from_record({"status": "ok", "final": "answer"}, mode="final"), "answer")


class TraceEvalOnlyContract(unittest.TestCase):
    """ADR 0004, item 6: the sample of the gate is evaluation data, so a trace made on one of its rows cannot be written (review of #78)."""

    OK = {"status": "ok", "truncated": False, "reasoning": "think", "final": "answer"}

    def gt_ids(self):
        manifest = json.loads((ROOT / "decide" / "think" / "gt-sample-manifest.json").read_text(encoding="utf-8"))
        return sorted(rid for b in manifest["benchmarks"].values() for rid in b["run_ids"])

    def test_the_ids_are_the_ones_of_the_committed_manifest(self) -> None:
        ids = self.gt_ids()
        self.assertEqual(sorted(TR.gt_sample_ids()), ids)
        self.assertGreater(len(ids), 300)

    def test_a_row_of_the_sample_gives_no_trace(self) -> None:
        for rid in (self.gt_ids()[0], self.gt_ids()[-1]):
            for mode in TR.MODES:
                with self.assertRaises(TR.EvaluationDataError) as cm:
                    TR.from_record({**self.OK, "id": rid}, mode=mode)
                self.assertIn("ADR 0004", str(cm.exception))
                self.assertIsInstance(cm.exception, TR.TraceError)

    def test_a_row_outside_the_sample_and_a_record_without_an_id_are_not_affected(self) -> None:
        self.assertEqual(TR.from_record({**self.OK, "id": "train:0001"}), "think")
        self.assertEqual(TR.from_record(self.OK), "think")

    def test_attaching_a_trace_to_a_request_of_the_sample_is_refused_when_the_source_is_named(self) -> None:
        rid = self.gt_ids()[3]
        with self.assertRaises(TR.EvaluationDataError):
            TR.attach_trace("state", "think", source_id=rid)
        self.assertEqual(TR.attach_trace("state", "think", source_id="train:1"), TR.attach_trace("state", "think"))
        self.assertEqual(TR.attach_trace("state", "", source_id=rid), "state", "no trace, nothing to refuse")

    def test_another_manifest_can_be_named(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            path.write_text(json.dumps({"benchmarks": {"25": {"run_ids": ["x:1", "x:2"]}, "58": {"run_ids": ["y:1"]}}}), encoding="utf-8")
            self.assertEqual(TR.gt_sample_ids(path), frozenset({"x:1", "x:2", "y:1"}))


class ServeContract(unittest.TestCase):
    """`vllm serve` for the thinker, on a rented or own GPU.  Not the Kaggle T4 (README, "Backends")."""

    # Flags found by READING the source of vllm 0.31.0 (engine/arg_utils.py, entrypoints/launchers/cli_args.py); not executed.
    KNOWN_FLAGS = {"--revision", "--host", "--port", "--served-model-name", "--max-model-len", "--gpu-memory-utilization",
                   "--max-num-seqs", "--dtype", "--reasoning-parser", "--default-chat-template-kwargs",
                   "--tensor-parallel-size", "--language-model-only", "--enforce-eager", "--download-dir"}

    def flags(self, argv):
        return [a for a in argv if a.startswith("--")]

    def test_default_command(self) -> None:
        cmd = SV.vllm_command()
        self.assertEqual(cmd[:3], ["vllm", "serve", C.QWEN_FP8_MODEL])
        self.assertEqual(cmd[cmd.index("--revision") + 1], C.QWEN_FP8_REVISION)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertEqual(cmd[cmd.index("--reasoning-parser") + 1], "qwen3")
        self.assertEqual(json.loads(cmd[cmd.index("--default-chat-template-kwargs") + 1]), {"enable_thinking": True})
        self.assertIn("--language-model-only", cmd)
        self.assertNotIn("--tensor-parallel-size", cmd)
        self.assertNotIn("--enforce-eager", cmd)

    def test_only_flags_the_pinned_vllm_defines(self) -> None:
        cmd = SV.vllm_command(tensor_parallel_size=2, enforce_eager=True, download_dir="/cache", language_model_only=True)
        self.assertEqual(set(self.flags(cmd)) - self.KNOWN_FLAGS, set())
        self.assertEqual(len(self.flags(cmd)), len(set(self.flags(cmd))))                  # no flag twice

    def test_revision_can_be_left_out_and_the_vision_tower_kept(self) -> None:
        cmd = SV.vllm_command(revision=None, language_model_only=False)
        self.assertNotIn("--revision", cmd)
        self.assertNotIn("--language-model-only", cmd)

    def test_thinking_off_is_the_other_json(self) -> None:
        cmd = SV.vllm_command(thinking=False)
        self.assertEqual(json.loads(cmd[cmd.index("--default-chat-template-kwargs") + 1]), {"enable_thinking": False})

    def test_extra_flags_go_last(self) -> None:
        cmd = SV.vllm_command(extra=["--enable-prefix-caching"])
        self.assertEqual(cmd[-1], "--enable-prefix-caching")

    def test_the_numbers_reach_the_flags(self) -> None:
        cmd = SV.vllm_command(port=9001, max_model_len=20000, gpu_memory_utilization=0.85, max_num_seqs=8, tensor_parallel_size=4)
        for flag, value in (("--port", "9001"), ("--max-model-len", "20000"), ("--gpu-memory-utilization", "0.85"),
                            ("--max-num-seqs", "8"), ("--tensor-parallel-size", "4")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value, flag)

    def test_the_server_is_never_exposed(self) -> None:
        for host in ("0.0.0.0", "10.0.0.5", "example.com", ""):
            with self.assertRaises(ValueError, msg=host):
                SV.vllm_command(host=host)
        for host in ("127.0.0.1", "localhost", "::1"):
            SV.vllm_command(host=host)

    def test_bad_numbers_are_refused(self) -> None:
        for bad in ({"gpu_memory_utilization": 0.0}, {"gpu_memory_utilization": 1.5}, {"max_model_len": 0},
                    {"max_num_seqs": 0}, {"tensor_parallel_size": 0}, {"port": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                SV.vllm_command(**bad)

    def test_no_key_on_the_command_line(self) -> None:
        self.assertFalse(any("key" in a.lower() for a in SV.vllm_command()))

    def test_the_vllm_pin_is_the_one_of_the_repository(self) -> None:
        text = (ROOT / "scripts" / "check_vllm_args.py").read_text(encoding="utf-8")
        self.assertIn(f'PINNED = "{SV.VLLM_VERSION}"', text)

    def test_print_gives_a_command_that_splits_back_to_argv(self) -> None:
        import contextlib
        import io
        import shlex
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = SV.main(["--print"])
        self.assertEqual(code, 0)
        self.assertEqual(shlex.split(buf.getvalue()), SV.vllm_command(C.QWEN_FP8_MODEL, C.QWEN_FP8_REVISION))

    def test_the_pinned_revision_follows_the_model(self) -> None:
        import contextlib
        import io
        import shlex
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            SV.main(["--print", "--model", C.QWEN_MODEL])
        cmd = shlex.split(buf.getvalue())
        self.assertEqual(cmd[2], C.QWEN_MODEL)
        self.assertEqual(cmd[cmd.index("--revision") + 1], C.QWEN_REVISION)

    def test_cli_refuses_a_public_host_with_exit_2(self) -> None:
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(SV.main(["--print", "--host", "0.0.0.0"]), 2)
        self.assertIn("loopback", err.getvalue())

    def test_run_replaces_the_process_with_the_command(self) -> None:
        import contextlib
        import io
        from unittest import mock
        with mock.patch.object(SV.os, "execvp") as ex, contextlib.redirect_stdout(io.StringIO()):
            SV.main(["--run"])
        ex.assert_called_once()
        self.assertEqual(ex.call_args[0][0], "vllm")
        self.assertEqual(ex.call_args[0][1], SV.vllm_command(C.QWEN_FP8_MODEL, C.QWEN_FP8_REVISION))

    @staticmethod
    def opener(script):
        import urllib.error
        state = {"i": 0, "seen": []}

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def call(req, timeout=None):
            state["seen"].append((req.full_url, req.get_header("Authorization")))
            out = script[min(state["i"], len(script) - 1)]
            state["i"] += 1
            if out == "ok":
                return Resp()
            if out == "refused":
                raise urllib.error.URLError("connection refused")
            raise urllib.error.HTTPError(req.full_url, out, "x", {}, None)
        call.state = state
        return call

    @staticmethod
    def fake_time():
        now = [0.0]
        return (lambda: now[0]), (lambda s: now.__setitem__(0, now[0] + s))

    def test_wait_ready_polls_until_the_server_answers(self) -> None:
        clock, sleep = self.fake_time()
        op = self.opener(["refused", "refused", 503, "ok"])
        self.assertTrue(SV.wait_ready("http://127.0.0.1:8000/v1/", 60, interval_s=5, opener=op, sleep=sleep, clock=clock))
        self.assertEqual(op.state["i"], 4)
        self.assertTrue(all(url == "http://127.0.0.1:8000/v1/models" for url, _ in op.state["seen"]))

    def test_wait_ready_gives_up_at_the_deadline(self) -> None:
        clock, sleep = self.fake_time()
        op = self.opener(["refused"])
        self.assertFalse(SV.wait_ready("http://h/v1", 20, interval_s=5, opener=op, sleep=sleep, clock=clock))
        self.assertEqual(op.state["i"], 5)                              # t = 0, 5, 10, 15, 20

    def test_a_401_means_the_server_is_up(self) -> None:
        clock, sleep = self.fake_time()
        self.assertTrue(SV.wait_ready("http://h/v1", 5, opener=self.opener([401]), sleep=sleep, clock=clock))

    def test_the_key_is_sent_as_a_bearer_token(self) -> None:
        clock, sleep = self.fake_time()
        op = self.opener(["ok"])
        SV.wait_ready("http://h/v1", 5, api_key="sekret", opener=op, sleep=sleep, clock=clock)
        self.assertEqual(op.state["seen"][0][1], "Bearer sekret")

    @unittest.skipUnless(importlib.util.find_spec("vllm"), "vllm is not installed: the flags were checked against its source only")
    def test_the_installed_vllm_accepts_the_command(self) -> None:
        spec = importlib.util.spec_from_file_location("check_vllm_args", ROOT / "scripts" / "check_vllm_args.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.validate(SV.vllm_command()[1:]), 0)


@unittest.skipIf(os.environ.get("THINK_MUTATION_CHECK") == "1", "inside a mutated copy this test fails for every mutant, which would hide the real one")
class MutationToolContract(unittest.TestCase):
    """decide/think/mutation_check.py plants one-line bugs in a copy and expects these tests to fail; its list must not go stale."""

    def test_every_mutant_still_applies_exactly_once(self) -> None:
        from decide.think import mutation_check as MC
        ids = [m.id for m in MC.MUTANTS]
        self.assertEqual(len(ids), len(set(ids)))
        for m in MC.MUTANTS:
            text = (ROOT / m.file).read_text(encoding="utf-8")
            self.assertEqual(text.count(m.old), 1, f"{m.id}: `old` must occur once in {m.file}")
            self.assertNotEqual(m.old, m.new, m.id)

    def test_the_groups_cover_the_places_that_decide_the_verdict(self) -> None:
        from decide.think import mutation_check as MC
        groups = {m.group for m in MC.MUTANTS}
        self.assertTrue({"answers", "bootstrap", "verdict", "scoring", "resume", "sample", "trace", "serve", "report", "hosted", "notebook"} <= groups)


class CommittedManifestContract(unittest.TestCase):
    """decide/think/gt-sample-manifest.json: the ids and hashes of the 400 rows drawn from the public suite (never their text)."""

    PATH = ROOT / "decide" / "think" / "gt-sample-manifest.json"

    @classmethod
    def setUpClass(cls) -> None:
        cls.m = json.loads(cls.PATH.read_text(encoding="utf-8"))

    def test_it_is_the_pre_registered_sample(self) -> None:
        m = self.m
        self.assertEqual((m["format"], m["seed"], m["per_benchmark"], m["one_per_case"]), (C.MANIFEST_FORMAT, C.SEED, C.PER_BENCHMARK, True))
        self.assertEqual(m["decision_models"], sorted(C.DECISION_MODELS))
        self.assertEqual(m["use"], C.EVAL_ONLY_NOTICE)

    def test_every_default_benchmark_is_drawn_or_listed_as_missing(self) -> None:
        drawn = {int(b) for b in self.m["benchmarks"]}
        self.assertEqual(drawn | set(self.m["missing_benchmarks"]), set(C.DEFAULT_BENCHMARKS))
        self.assertEqual(drawn & set(self.m["missing_benchmarks"]), set())
        self.assertEqual(self.m["missing_benchmarks"], [45], "HLE is gated on the Hub: the rebuilt suite has none of its rows")

    def test_counts_ids_and_hashes(self) -> None:
        hexes = re.compile(r"^[0-9a-f]{64}$")
        total = 0
        seen = set()
        for b, e in self.m["benchmarks"].items():
            self.assertEqual(e["n"], C.PER_BENCHMARK, b)
            self.assertEqual((len(e["run_ids"]), len(e["payload_sha256"])), (e["n"], e["n"]), b)
            self.assertEqual(e["name"], C.benchmark_name(int(b)))
            self.assertTrue(all(hexes.match(h) for h in e["payload_sha256"]), b)
            self.assertTrue(all(r.startswith(f"{b}:") or r.startswith(("v0.3-gsm8k:", "candidates-v3:", "calibration-v1:")) for r in e["run_ids"]), b)
            self.assertEqual(e["skipped_ineligible_cases"], 0)
            total += e["n"]
            seen.update(e["run_ids"])
        self.assertEqual(len(seen), total, "a request id appears twice")
        self.assertEqual(self.m["pack"]["rows"], total)
        self.assertTrue(hexes.match(self.m["pack"]["content_sha256"]))

    def test_inputs_are_named_by_sha256(self) -> None:
        hexes = re.compile(r"^[0-9a-f]{64}$")
        self.assertEqual(sorted(self.m["inputs"]["results"]), sorted(C.DECISION_MODELS))
        for group in self.m["inputs"].values():
            for name, digests in group.items():
                self.assertTrue(all(hexes.match(v) for v in digests.values()), name)
                self.assertIn("sha256", digests)
        pinned = {"added-rows.jsonl.gz": "7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89",
                  "gsm8k-rows.jsonl.gz": "759334894858a37d6bd7bd07298eac8e55da236e31dad327ccd06a9979f4738a"}     # the kit's own pins for edition 0.3
        for name, digest in pinned.items():
            self.assertEqual(self.m["inputs"]["suite_files"][name]["uncompressed_sha256"], digest, name)

    def test_the_file_holds_no_question_text(self) -> None:
        self.assertEqual(sorted(self.m), ["benchmarks", "decision_models", "format", "inputs", "missing_benchmarks", "note", "one_per_case",
                                          "order", "pack", "per_benchmark", "seed", "skipped", "use"])
        for e in self.m["benchmarks"].values():
            self.assertEqual(sorted(e), ["n", "name", "payload_sha256", "population_cases", "run_ids", "skipped_ineligible_cases"])
        self.assertLess(self.PATH.stat().st_size, 200_000)


class ProvenanceContract(unittest.TestCase):
    """Which model and revision the results are credited to: declared, never guessed for an endpoint."""

    @staticmethod
    def config(*argv):
        args = T.build_parser().parse_args(["--pack", "p", "--out", "o", *argv])
        return T.make_config(args)

    def test_the_pins_are_in_one_table(self) -> None:
        self.assertEqual(C.PINNED_REVISIONS, {C.QWEN_MODEL: C.QWEN_REVISION, C.QWEN_FP8_MODEL: C.QWEN_FP8_REVISION})
        for rev in C.PINNED_REVISIONS.values():
            self.assertRegex(rev, r"^[0-9a-f]{40}$")

    def test_transformers_defaults_to_the_pinned_revision_of_its_model(self) -> None:
        self.assertEqual(self.config("--backend", "transformers")["revision"], C.QWEN_REVISION)
        self.assertEqual(self.config("--backend", "transformers", "--model", C.QWEN_FP8_MODEL)["revision"], C.QWEN_FP8_REVISION)
        self.assertIsNone(self.config("--backend", "transformers", "--model", "someone/else")["revision"])
        self.assertIsNone(self.config("--backend", "transformers", "--revision", "none")["revision"])
        self.assertEqual(self.config("--backend", "transformers", "--revision", "abc")["revision"], "abc")

    def test_an_endpoint_serves_what_it_serves_so_the_revision_is_only_what_you_declare(self) -> None:
        self.assertIsNone(self.config("--backend", "openai")["revision"])
        self.assertIsNone(self.config("--backend", "openai", "--model", C.QWEN_FP8_MODEL)["revision"])
        self.assertEqual(self.config("--backend", "openai", "--model", C.QWEN_FP8_MODEL, "--revision", C.QWEN_FP8_REVISION)["revision"], C.QWEN_FP8_REVISION)

    def test_the_compute_dtype_is_part_of_the_protocol_and_the_quantization_label_follows_it(self) -> None:
        base = self.config("--backend", "transformers")
        self.assertEqual((base["dtype"], base["quantization"]), ("float16", "nf4-fp16"))
        for flag, short in (("bfloat16", "bf16"), ("float32", "fp32")):
            cfg = self.config("--backend", "transformers", "--dtype", flag)
            self.assertEqual((cfg["dtype"], cfg["quantization"]), (flag, f"nf4-{short}"))
            self.assertNotEqual(T.fingerprint_of(cfg), T.fingerprint_of(base), flag)
        plain = self.config("--backend", "transformers", "--quant", "none", "--dtype", "bfloat16")
        self.assertEqual((plain["quantization"], plain["dtype"]), ("none", "bfloat16"))
        api = self.config("--backend", "openai")
        self.assertEqual((api["quantization"], api["dtype"]), (None, None), "an endpoint has no dtype of ours to declare")

    def test_the_nf4_matmuls_run_in_the_dtype_that_was_chosen(self) -> None:
        marker = object()
        kw = T.nf4_config_kwargs(marker)
        self.assertIs(kw["bnb_4bit_compute_dtype"], marker)
        self.assertEqual((kw["load_in_4bit"], kw["bnb_4bit_quant_type"], kw["bnb_4bit_use_double_quant"]), (True, "nf4", True))
        src = inspect.getsource(T.TransformersBackend.__init__)
        self.assertIn("nf4_config_kwargs(torch_dtype)", src, "the backend must hand the chosen dtype to the NF4 config, not a literal fp16")
        self.assertNotIn("bnb_4bit_compute_dtype=torch.float16", src)

    def test_serve_names_the_model_as_the_thinker_asks_for_it(self) -> None:
        self.assertEqual(SV.vllm_command().count("--served-model-name"), 1)
        cmd = SV.vllm_command()
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], C.QWEN_FP8_MODEL)
        cmd = SV.vllm_command(C.QWEN_MODEL, C.QWEN_REVISION)
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], C.QWEN_MODEL)
        cmd = SV.vllm_command(served_name="mine")
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "mine")

    def test_the_report_accepts_both_pinned_pairs(self) -> None:
        for model, rev in C.PINNED_REVISIONS.items():
            md = ThinkerConfigInReportContract.md(model=model, revision=rev)
            self.assertNotIn("AVISO: modelo ou revisao", md, model)


class GarbledOutputContract(unittest.TestCase):
    """fp16 can overflow on a T4 and make the model emit noise; that must not read as a quiet FAIL."""

    GOOD = "Let us compute the energy first: E = h * c / lambda, so the photon of 500 nm carries about 2.48 eV. Therefore option B is the closest. " * 4

    def test_text_that_reads_is_not_garbled(self) -> None:
        self.assertFalse(K.looks_garbled(self.GOOD))
        self.assertFalse(K.looks_garbled("Resolvendo: \\frac{a}{b} = 3 \\Rightarrow a = 3b; assim 2a + b = 7b. " * 5))      # LaTeX is symbol-heavy but still readable

    def test_a_run_of_one_character_is_garbled(self) -> None:
        self.assertTrue(K.looks_garbled("!" * 500))
        self.assertTrue(K.looks_garbled("�" * 300))

    def test_mostly_symbols_is_garbled(self) -> None:
        self.assertTrue(K.looks_garbled("#$%&*()+=<>?@^~|" * 40))

    def test_short_or_empty_text_is_not_judged(self) -> None:
        for text in ("", None, "!!!", "x" * 100):
            self.assertFalse(K.looks_garbled(text), repr(text))

    def test_only_the_first_2000_characters_are_read(self) -> None:
        self.assertFalse(K.looks_garbled(self.GOOD * 5 + "!" * 20000))              # the noise after the window is not read

    def test_stats_count_the_garbled_rows_from_reasoning_or_final(self) -> None:
        rows, recs = scenario({25: [(True, {"m": True})] * 4})
        recs[0]["reasoning"] = self.GOOD
        recs[1]["reasoning"] = "!" * 400
        recs[2]["final"] = "�" * 400
        stats = K.thinker_stats(K.score_pack(rows, recs))
        self.assertEqual(stats["garbled"], 2)

    def test_the_report_warns_when_any_row_is_garbled(self) -> None:
        rows, recs = scenario({25: [(False, {"m": True})] * 5})
        for r in recs[:3]:
            r["reasoning"] = "!" * 400
        md = K.render_markdown(K.build_report(rows, recs, B=20, seed=1))
        self.assertIn("AVISO: 3 linhas com saida ilegivel", md)
        self.assertIn("fp16", md)
        quiet = K.render_markdown(K.build_report(*scenario({25: [(False, {"m": True})] * 5}), B=20, seed=1))
        self.assertNotIn("saida ilegivel", quiet)


class ExtraBodySecretContract(unittest.TestCase):
    """--extra-body is copied into the results header and the report, which get pasted into issues: a key must not get in."""

    KEY = "sk-abcdefghijklmnopqrstuvwxyz0123"

    def refused(self, text):
        with self.assertRaises(SystemExit) as cm:
            T._json_arg(text, "--extra-body")
        message = str(cm.exception)
        self.assertIn("THINK_API_KEY", message)
        self.assertNotIn(self.KEY, message, "the refusal must not print the secret it refuses")
        return message

    def test_a_key_named_like_a_secret_is_refused_at_any_depth(self) -> None:
        for name in ("api_key", "API-KEY", "apiKey", "Authorization", "password", "client_secret", "access_token", "token"):
            self.assertIn(name, self.refused(json.dumps({"a": {"b": [{name: "x"}]}})), name)

    def test_a_value_shaped_like_a_key_is_refused_wherever_it_sits(self) -> None:
        for value in (self.KEY, "Bearer abcdefghijkl", "hf_" + "A" * 30):
            self.refused(json.dumps({"headers": [{"note": value}]}))

    def test_the_fields_a_provider_really_takes_pass(self) -> None:
        for obj in ({"enable_thinking": True}, {"thinking": {"type": "enabled", "budget_tokens": 4000}}, {"max_new_tokens": 5},
                    {"reasoning": {"effort": "high"}}, {"token_limit": 3}, {"thinking_budget": 100}, {"note": "Bearer"}, {"id": "sk-short"},
                    {"provider": {"order": ["a"], "allow_fallbacks": False}}):
            self.assertEqual(T._json_arg(json.dumps(obj), "--extra-body"), obj)

    def test_the_cli_stops_before_a_results_file_exists(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            pack, out = Path(d) / "pack.jsonl.gz", Path(d) / "r.jsonl"
            C.write_jsonl_gz(pack, [prow(25, 0, picks={"m": "A"})])
            with self.assertRaises(SystemExit):
                T.main(["--pack", str(pack), "--out", str(out), "--backend", "openai", "--base-url", "http://127.0.0.1:1/v1",
                        "--extra-body", json.dumps({"api_key": self.KEY})])
            self.assertFalse(out.exists())


class ExtraBodyProtocolContract(unittest.TestCase):
    """--extra-body is for provider fields.  The fields that make the protocol have their own options, and the request is
    built so that the extra body cannot override them (review of #78, blocker 2a: a request for no thinking, temperature 0
    and 64 tokens went out while the header and the report said thinking on, 16384 tokens, temperature 1)."""

    THE_REVIEWERS = '{"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"max_tokens":64}'

    def test_the_request_of_the_review_is_refused_before_anything_is_written(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            T._json_arg(self.THE_REVIEWERS, "--extra-body")
        self.assertIn("chat_template_kwargs", str(cm.exception))

    def test_every_protocol_field_is_refused_by_name_whatever_its_case(self) -> None:
        for key in sorted(C.PROTOCOL_KEYS) + ["Temperature", "MAX_TOKENS"]:
            with self.assertRaises(SystemExit) as cm:
                T._json_arg(json.dumps({"ok": 1, key: 1}), "--extra-body")
            self.assertIn(key, str(cm.exception), key)
            self.assertIn("protocol", str(cm.exception), key)

    def test_the_protocol_fields_cover_the_sampling_the_template_the_budget_and_the_stream(self) -> None:
        self.assertTrue(set(C.THINKING_SAMPLING) <= C.PROTOCOL_KEYS)
        for key in ("model", "messages", "max_tokens", "max_completion_tokens", "seed", "stream", "stream_options", "chat_template_kwargs", "reasoning_effort"):
            self.assertIn(key, C.PROTOCOL_KEYS)

    def test_a_nested_key_with_a_protocol_name_is_a_provider_field_not_a_protocol_field(self) -> None:
        obj = {"provider": {"max_tokens": 5}}
        self.assertEqual(T._json_arg(json.dumps(obj), "--extra-body"), obj)

    def test_the_backend_refuses_them_too_for_whoever_builds_it_without_the_cli(self) -> None:
        for key in ("temperature", "max_tokens", "chat_template_kwargs", "seed", "stream", "Temperature", "MAX_TOKENS"):
            with self.assertRaises(T.BackendError) as cm:
                T.OpenAIBackend(base_url="http://127.0.0.1:1/v1", model="m", extra_body={key: 1})
            self.assertIn(key, str(cm.exception))

    def test_a_results_header_that_says_the_extra_body_set_a_protocol_field_is_not_the_gate(self) -> None:
        self.assertEqual(K.protocol_keys_in(fake_config(extra_body={"temperature": 0, "x": 1})), ["temperature"])
        self.assertEqual(K.protocol_keys_in(fake_config(extra_body={"x": 1})), [])
        self.assertEqual(K.protocol_keys_in(None), [])


class KeySafetyContract(unittest.TestCase):
    """The API key goes to the endpoint the user named, over a channel that keeps it, and nowhere else (review of #78, blocker 3)."""

    KEY = "sk-test-secret-123"

    def make(self, url, key=KEY):
        return T.OpenAIBackend(base_url=url, model="m", api_key=key)

    def test_a_key_is_not_sent_in_clear_to_another_machine(self) -> None:
        for url in ("http://api.example.com/v1", "http://10.0.0.5:8000/v1", "http://[2001:db8::1]:8000/v1", "http://localhost.example.com/v1",
                    "http://127.0.0.1.example.com/v1", "http://0.0.0.0:8000/v1", "HTTP://API.EXAMPLE.COM/v1"):
            with self.assertRaises(T.BackendError) as cm:
                self.make(url)
            self.assertIn("https://", str(cm.exception), url)
            self.assertNotIn(self.KEY, str(cm.exception))

    def test_a_key_may_go_in_clear_to_this_machine_and_over_tls_anywhere(self) -> None:
        for url in ("http://127.0.0.1:8000/v1", "http://localhost:8000/v1", "http://127.1.2.3/v1", "http://[::1]:8000/v1", "https://api.example.com/v1",
                    "HTTPS://api.example.com/v1", "http://LOCALHOST:8000/v1"):
            self.make(url)

    def test_without_a_key_any_host_goes(self) -> None:
        for url in ("http://api.example.com/v1", "http://10.0.0.5:8000/v1"):
            self.make(url, key=None)
            self.make(url, key="")

    def test_only_http_and_https_are_endpoints(self) -> None:
        for url in ("file:///etc/passwd", "ftp://host/v1", "gopher://host/", "//host/v1", "host:8000/v1"):
            with self.assertRaises(T.BackendError, msg=url):
                self.make(url, key=None)

    def test_a_redirect_is_never_followed(self) -> None:
        import urllib.request
        handler = T._NoRedirect()
        req = urllib.request.Request("https://a.example/v1/chat/completions", data=b"{}", headers={"Authorization": "Bearer x"}, method="POST")
        for code in (301, 302, 303, 307, 308):
            self.assertIsNone(handler.redirect_request(req, None, code, "moved", {}, "https://evil.example/"), code)

    def test_the_backend_opens_with_the_opener_that_does_not_redirect(self) -> None:
        self.assertIs(self.make("http://127.0.0.1:1/v1").opener, T.open_no_redirect)
        self.assertIs(T.OpenAIBackend.__init__.__kwdefaults__["opener"], T.open_no_redirect)

    def test_a_redirect_stops_the_run_and_says_where_it_points_without_following_it(self) -> None:
        hops = []

        def opener(req, timeout):
            hops.append(req.full_url)
            raise http_error(302, "", {"Location": "https://evil.example/steal?token=abc"})
        b = T.OpenAIBackend(base_url="http://127.0.0.1:1/v1", model="m", api_key=self.KEY, opener=opener, sleep=lambda s: None, stream=False)
        with self.assertRaises(T.BackendError) as cm:
            b.generate([T.Item("a", "p")])
        self.assertEqual(len(hops), 1)
        msg = str(cm.exception)
        self.assertIn("redirect", msg)
        self.assertIn("https://evil.example/steal", msg)
        self.assertNotIn("token=abc", msg, "the query of a Location can hold a credential")
        self.assertNotIn(self.KEY, msg)
        with self.assertRaises(T.BackendError) as cm:
            b.preflight()
        self.assertIn("redirect", str(cm.exception))

    def test_the_cli_refuses_a_key_over_plain_http_to_another_machine_before_a_results_file_exists(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            pack, out = Path(d) / "pack.jsonl.gz", Path(d) / "r.jsonl"
            C.write_jsonl_gz(pack, [prow(25, 0, picks={"m": "A"})])
            with mock.patch.dict(os.environ, {"THINK_API_KEY": self.KEY}), contextlib.redirect_stdout(io.StringIO()) as buf:
                code = T.main(["--pack", str(pack), "--out", str(out), "--backend", "openai", "--base-url", "http://api.example.com/v1", "--no-preflight"])
            self.assertEqual(code, 2)
            self.assertIn("https://", buf.getvalue())
            self.assertNotIn(self.KEY, buf.getvalue())


class NoThinkingContract(unittest.TestCase):
    """An endpoint that ignores the request for thinking answers with a bare line: say so, do not score it as a quiet FAIL."""

    @staticmethod
    def md(tokens, **cfg):
        rows, recs = scenario({25: [(False, {"m": True})] * 5})
        for r in recs:
            r["tokens"] = {"completion": tokens}
        header = {"config": fake_config(model=C.QWEN_MODEL, revision=C.QWEN_REVISION, backend="openai", **cfg), "config_fingerprint": "f"}
        return K.render_markdown(K.build_report(rows, recs, header=header, B=20, seed=1))

    def test_the_threshold_is_one_constant(self) -> None:
        self.assertEqual(C.MIN_THINKING_TOKENS, 100)

    def test_a_median_below_it_warns_and_exactly_at_it_does_not(self) -> None:
        self.assertIn("AVISO: a mediana e de 99 tokens por linha: o modelo quase nao pensou", self.md(99))
        self.assertNotIn("quase nao pensou", self.md(C.MIN_THINKING_TOKENS))
        self.assertNotIn("quase nao pensou", self.md(5000))

    def test_the_warning_points_at_the_fix(self) -> None:
        md = self.md(10)
        self.assertIn("Isto nao e o gate GT", md)
        self.assertIn("outro `--out`", md)

    def test_a_thinking_off_control_is_expected_to_be_terse(self) -> None:
        self.assertNotIn("quase nao pensou", self.md(12, chat_template_kwargs={"enable_thinking": False}))

    def test_rows_without_token_counts_are_not_guessed_about(self) -> None:
        rows, recs = scenario({25: [(False, {"m": True})] * 5})
        for r in recs:
            del r["tokens"]
        header = {"config": fake_config(model=C.QWEN_MODEL, revision=C.QWEN_REVISION), "config_fingerprint": "f"}
        self.assertNotIn("quase nao pensou", K.render_markdown(K.build_report(rows, recs, header=header, B=20, seed=1)))

    def test_a_results_file_without_a_header_is_read_as_a_thinking_run(self) -> None:
        rows, recs = scenario({25: [(False, {"m": True})] * 5})
        self.assertIn("quase nao pensou", K.render_markdown(K.build_report(rows, recs, B=20, seed=1)))

    def store(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return T.ResultStore.open(Path(tmp.name) / "r.jsonl", fake_config(), "p" * 64, 5)

    def test_the_thinker_summary_warns_on_a_short_average_and_only_for_a_thinking_run(self) -> None:
        store = self.store()
        short = T.RunStats(total=3, done=3, ok=3, completion_tokens=3 * 99)
        self.assertIn("WARNING: 99 tokens per reply on average: the model barely thought", T.summarize(short, store))
        self.assertNotIn("barely thought", T.summarize(short, store, thinking=False))
        enough = T.RunStats(total=3, done=3, ok=3, completion_tokens=3 * C.MIN_THINKING_TOKENS)
        self.assertNotIn("barely thought", T.summarize(enough, store))
        nothing_ok = T.RunStats(total=3, done=3, errors=3)
        self.assertNotIn("barely thought", T.summarize(nothing_ok, store), "no ok row, no average (and no division by zero)")

    def test_the_thinker_summary_counts_replies_without_thinking_and_unclosed_ones_and_rows_given_up(self) -> None:
        store = self.store()
        many = C.MIN_THINKING_TOKENS * 5
        quiet = T.RunStats(total=5, done=5, ok=5, completion_tokens=5 * many, no_thinking=2, no_close=1)
        text = T.summarize(quiet, store)
        self.assertIn("2 of 5 replies show no thinking", text)
        self.assertIn("1 of 5 replies stopped without closing the thinking", text)
        clean = T.RunStats(total=5, done=5, ok=5, completion_tokens=5 * many)
        self.assertNotIn("show no thinking", T.summarize(clean, store))
        self.assertNotIn("without closing", T.summarize(clean, store))
        self.assertNotIn("show no thinking", T.summarize(quiet, store, thinking=False), "a control run is not asked to think")
        self.assertNotIn("without closing", T.summarize(quiet, store, thinking=False))
        store.append({"id": "a", "kind": "row", "status": "failed", "gave_up": 3})
        store.append({"id": "b", "kind": "row", "status": "failed"})
        store.append({"id": "c", "kind": "row", "status": "error"})
        self.assertIn("2 failed (1 given up after repeated errors), 1 to retry", T.summarize(clean, store))


# ------------------------------------------------------------------------------------------- notebooks

def parse_cli(parser, argv):
    """Parse the way the CLI does, but fail with argparse's own message instead of exiting."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return parser.parse_args(argv)
    except SystemExit as e:
        raise AssertionError(f"argparse refused {argv}: {err.getvalue().strip().splitlines()[-1:]}") from e


CLI_PARSERS = {"decide.think.thinker": T.build_parser, "decide.think.compare": K.build_parser, "decide.think.sample": S.build_parser}


def parsed_commands(key, **params):
    """({variable: value} of the parameter cell, {thinker|compare|sample: Namespace}) with the real parsers."""
    ns = NB.parameters(key, **params)
    out = {}
    for module, build in CLI_PARSERS.items():
        for cmd in NB.commands_of(key, module):
            argv = NB.argv_of(cmd, ns)
            out[module.rsplit(".", 1)[1]] = parse_cli(build(), argv[3:])           # [python, -m, module, ...]
    return ns, out


class NotebookFileContract(unittest.TestCase):
    """notebooks/decide/think_gt_*.ipynb: the two ways to run the gate (issue #75).  Read as data; no cell runs here."""

    KEYS = tuple(NB.NOTEBOOKS)

    def test_the_two_notebooks_are_where_the_layout_rule_allows(self) -> None:
        self.assertEqual(sorted(p.name for p in NB.NB_DIR.glob("*.ipynb") if p.name.startswith("think_gt_")),
                         ["think_gt_api.ipynb", "think_gt_kaggle.ipynb"])
        self.assertEqual(NB.NB_DIR, ROOT / "notebooks" / "decide")

    def test_nbformat_4_and_nothing_left_by_a_run(self) -> None:
        for key in self.KEYS:
            nb = NB.load(key)
            self.assertEqual(nb["nbformat"], 4, key)
            ids = []
            for cell in nb["cells"]:
                self.assertIn(cell["cell_type"], ("markdown", "code"), key)
                self.assertTrue(isinstance(cell["source"], list) and all(isinstance(s, str) for s in cell["source"]), key)
                ids.append(cell.get("id"))
                if cell["cell_type"] == "code":
                    self.assertEqual((cell["outputs"], cell["execution_count"]), ([], None), f"{key}: a run left its output in the file")
            self.assertNotIn(None, ids, key)
            self.assertEqual(len(set(ids)), len(ids), f"{key}: cell ids repeat")

    def test_no_secret_no_wildcard_bind_no_picture(self) -> None:
        for key, path in NB.NOTEBOOKS.items():
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{20,}|Bearer\s", key)
            self.assertNotIn("0.0.0.0", text, key)
            self.assertNotIn("image/png", text, key)

    def test_the_first_code_cell_is_the_one_the_user_edits(self) -> None:
        for key in self.KEYS:
            first = NB.code_cells(key)[0]
            self.assertTrue(first.startswith("# 1. Parâmetros"), key)
            for name in ("SIMPLICIO_SHA", "PACK", "WORK", "MAX_TOKENS", "LIMIT", "EXTRA"):
                self.assertRegex(first, rf"(?m)^{name}\s*=", f"{key}: {name}")

    def test_the_shell_lines_use_only_quoted_strings_numbers_and_the_option_strings_built_with_quotes(self) -> None:
        """`!cmd {name}` pastes the value into a shell line.  A string the user types goes through `q(...)` (shlex.quote), a number
        goes as it is, and the two option strings (EXTRA, REV) are built in Python from quoted parts (review of #78: an `&` in a URL cut the command)."""
        built = {"EXTRA", "REV"}
        for key in self.KEYS:
            ns = NB.parameters(key)
            ns.update({"BASE_URL": "u", "REV": ""})
            for _, cmd in NB.shell_commands(key):
                self.assertNotIn("$", cmd, f"{key}: IPython would read $name as a Python variable: {cmd}")
                for expr in NB.INTERPOLATION.findall(cmd):
                    expr = expr.strip()
                    quoted = re.fullmatch(r"q\(\s*([A-Za-z_]\w*)\s*\)", expr)
                    name = quoted.group(1) if quoted else expr
                    self.assertIn(name, ns, f"{key}: {{{expr}}} is not a variable of the parameter cell")
                    if quoted:
                        continue
                    self.assertTrue(isinstance(ns[name], (int, float)) or name in built,
                                    f"{key}: {{{expr}}} pastes a string into the shell unquoted: use {{q({name})}}")
            self.assertIn("from shlex import quote as q", NB.code_cells(key)[0], key)

    def test_a_value_with_a_space_a_quote_an_ampersand_or_a_substitution_reaches_the_program_as_one_argument(self) -> None:
        hostile = "a b&c;d$(touch x)`id`'e\"f"
        for key in self.KEYS:
            params = {"PACK": hostile, "WORK": hostile + "w"}
            if key == "api":
                params.update(BASE_URL=hostile + "u", MODEL=hostile + "m", REVISION=hostile + "r", LIMIT=hostile + "l", STANDARD_ONLY=True)
            else:
                params.update(LIMIT=hostile + "l")
            ns = NB.parameters(key, **params)
            ns.setdefault("BASE_URL", hostile + "u")
            for module, flags in (("decide.think.thinker", ["--pack", "--out", "--manifest", "--limit"] + (["--base-url", "--model", "--revision"] if key == "api" else [])),
                                  ("decide.think.compare", ["--pack", "--thinker", "--out"]), ("decide.think.sample", ["--pack"])):
                for cmd in NB.commands_of(key, module):
                    argv = NB.argv_of(cmd, ns)
                    for flag in flags:
                        if flag in argv:
                            value = argv[argv.index(flag) + 1]
                            expected = {"--pack": hostile, "--out": ns["WORK"] + ("/results.jsonl" if module.endswith("thinker") else "/report"),
                                        "--thinker": ns["WORK"] + "/results.jsonl", "--manifest": NB.MANIFEST_PATH, "--limit": hostile + "l",
                                        "--base-url": hostile + "u", "--model": hostile + "m", "--revision": hostile + "r"}[flag]
                            self.assertEqual(value, expected, f"{key} {module} {flag}")
            self.assertTrue(all("touch" not in a or a.startswith("a b&c") for cmd in NB.commands_of(key, "decide.think.thinker") for a in NB.argv_of(cmd, ns)))

    def test_the_python_of_the_cells_has_no_undefined_name_or_syntax_error(self) -> None:
        ruff = shutil.which("ruff")
        if ruff is None:
            self.skipTest("ruff is not on the PATH (make check has it)")
        for key in self.KEYS:
            r = subprocess.run([ruff, "check", "--no-cache", "--select", "F,E9", "--stdin-filename", f"{key}.py", "-"],
                               input=NB.python_source(key), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{key}:\n{r.stdout}{r.stderr}")


class NotebookPinContract(unittest.TestCase):
    KEYS = tuple(NB.NOTEBOOKS)

    @staticmethod
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    def test_both_notebooks_pin_the_same_full_commit(self) -> None:
        shas = {key: NB.pinned_sha(key) for key in self.KEYS}
        for key, sha in shas.items():
            self.assertRegex(sha, r"^[0-9a-f]{40}$", key)
        self.assertEqual(len(set(shas.values())), 1, shas)

    def test_the_code_is_fetched_by_that_commit_and_checked(self) -> None:
        for key in self.KEYS:
            code = "\n".join(NB.code_cells(key))
            self.assertIn('"fetch", "-q", "--depth", "1", "https://github.com/simpletibr/simplicio-27b.git", SIMPLICIO_SHA', code, key)
            self.assertIn('["git", "-C", REPO, "checkout", "-q", "FETCH_HEAD"]', code, key)
            self.assertIn("assert head == SIMPLICIO_SHA", code, key)
            self.assertNotRegex(code, r"git clone|releases/latest|raw\.githubusercontent|refs/heads|\bmain\b|\bmaster\b", key)

    def test_the_pinned_commit_holds_the_code_these_tests_cover(self) -> None:
        """What the user runs is what was tested: every module and the manifest are the same blobs at the pinned commit.

        A clone that lacks the commit, or whose history does not hold it, FAILS (it used to skip, and the guard against drift went quiet)."""
        sha = NB.pinned_sha("api")
        kind, message = NB.pin_state(ROOT, sha)
        if kind == "not_git":
            self.skipTest(message)                       # a source archive has no history to compare with
        self.assertEqual(kind, "ok", message)
        wanted = re.compile(r"^decide/think/(?:[A-Za-z0-9_]+\.py|gt-sample-manifest\.json)$")
        here = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "decide" / "think").iterdir() if wanted.match(str(p.relative_to(ROOT))))
        there = sorted(f for f in self.git("ls-tree", "-r", "--name-only", sha, "decide/think").stdout.split() if wanted.match(f))
        self.assertEqual(there, here, "a module was added or removed after the commit the notebooks pin")
        stale = [rel for rel in here
                 if self.git("rev-parse", f"{sha}:{rel}").stdout.strip() != self.git("hash-object", rel).stdout.strip()]
        self.assertEqual(stale, [], "these changed after the commit the notebooks pin: change SIMPLICIO_SHA in both notebooks "
                                   "to a pushed commit that has them")

    def test_a_pin_that_is_not_in_the_history_of_the_checkout_fails_with_the_way_to_get_it(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = C.clean_git_env()

            def git(*args, cwd=None):
                r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(cwd or root / "repo"), *args],
                                   capture_output=True, text=True, env=env)
                self.assertEqual(r.returncode, 0, r.stderr)
                return r.stdout.strip()
            (root / "repo").mkdir()
            git("init", "-q", "-b", "main")
            for name in ("a", "b", "c"):
                git("commit", "-q", "--allow-empty", "-m", name)
            first, second = git("rev-parse", "HEAD~2"), git("rev-parse", "HEAD~1")
            git("checkout", "-q", "-b", "side", second)
            git("commit", "-q", "--allow-empty", "-m", "squashed away")
            side = git("rev-parse", "HEAD")
            git("checkout", "-q", "main")
            repo = root / "repo"
            self.assertEqual(NB.pin_state(repo, first, env), ("ok", ""))
            kind, message = NB.pin_state(repo, "0" * 40, env)
            self.assertEqual(kind, "missing")
            self.assertIn("git fetch --unshallow", message)
            self.assertIn(f"git fetch origin {'0' * 40}", message)
            self.assertNotIn("it is a shallow clone", message, "this clone is complete: the message must not blame a shallow one")
            kind, message = NB.pin_state(repo, side, env)                                  # in the object store, not in the history of HEAD
            self.assertEqual(kind, "not_ancestor")
            self.assertIn("merge commit", message)
            self.assertIn("SIMPLICIO_SHA", message)
            shallow = root / "shallow"
            git("clone", "-q", "--depth", "1", repo.resolve().as_uri(), str(shallow), cwd=root)
            kind, message = NB.pin_state(shallow, first, env)
            self.assertEqual(kind, "missing", "a shallow clone has no parent commits")
            self.assertIn("(it is a shallow clone)", message)
            plain = root / "plain"
            plain.mkdir()
            self.assertEqual(NB.pin_state(plain, first, {**env, "GIT_CEILING_DIRECTORIES": str(root)})[0], "not_git")


    def test_kaggle_installs_the_four_pins_of_the_colab_notebooks(self) -> None:
        installs = [c for _, c in NB.shell_commands("kaggle") if re.match(r"pip install\b", c)]
        self.assertEqual(len(installs), 1, installs)
        names = [t for t in shlex.split(installs[0]) if not t.startswith("-") and t not in ("pip", "install")]
        self.assertEqual(names, ["transformers==5.5.0", "accelerate==1.15.0", "bitsandbytes==0.50.2", "huggingface_hub==1.33.0"])
        train = (ROOT / "notebooks/Simplicio_27B_Training_Colab.ipynb").read_text(encoding="utf-8")
        for pin in names:
            self.assertIn(pin, train, f"{pin} is not a pin of the Colab Training notebook")

    def test_vllm_only_in_the_api_notebook_and_only_at_the_repository_pin(self) -> None:
        self.assertEqual(SV.VLLM_VERSION, "0.31.0")
        api = "\n".join(NB.code_cells("api"))
        self.assertEqual(re.findall(r"vllm==[0-9][0-9.]*", api), [f"vllm=={SV.VLLM_VERSION}"])
        self.assertNotRegex("\n".join(NB.code_cells("kaggle")), r"(?i)vllm", "a T4 cannot run the pinned vLLM (README, Backends)")
        for key in self.KEYS:
            self.assertEqual([c for _, c in NB.shell_commands(key) if re.search(r"pip install.*(?<![=\w])vllm(?!==)", c)], [], key)


class GitEnvContract(unittest.TestCase):
    """A test or a tool that runs git in a throwaway directory must not act on the repository that is being pushed
    (a pre-push hook in a linked worktree exports GIT_DIR)."""

    def test_the_variables_that_name_a_repository_are_dropped_and_the_rest_kept(self) -> None:
        env = {"GIT_DIR": "/x", "GIT_WORK_TREE": "/y", "GIT_INDEX_FILE": "/z", "GIT_COMMON_DIR": "/c", "GIT_PREFIX": "p", "GIT_CONFIG_KEY_0": "a",
               "GIT_CONFIG_VALUE_0": "b", "GIT_CONFIG_COUNT": "1", "HOME": "/h", "PATH": "/bin", "GIT_AUTHOR_NAME": "n"}
        self.assertEqual(C.without_local_git_vars(env), {"HOME": "/h", "PATH": "/bin", "GIT_AUTHOR_NAME": "n"})

    def test_it_covers_every_variable_git_itself_lists(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        r = subprocess.run(["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertLessEqual(set(r.stdout.split()), set(C.LOCAL_GIT_VARS))

    def test_the_clean_environment_reads_no_user_config_and_never_asks_for_a_password(self) -> None:
        with mock.patch.dict(os.environ, {"GIT_DIR": "/x", "HOME": "/h"}):
            env = C.clean_git_env()
        self.assertNotIn("GIT_DIR", env)
        self.assertEqual((env["GIT_CONFIG_GLOBAL"], env["GIT_CONFIG_SYSTEM"], env["GIT_TERMINAL_PROMPT"]), (os.devnull, os.devnull, "0"))

    def test_the_mutation_tool_hands_the_copy_an_environment_without_them(self) -> None:
        from decide.think import mutation_check as M
        with mock.patch.dict(os.environ, {"GIT_DIR": "/x", "GIT_INDEX_FILE": "/y", "GIT_COMMON_DIR": "/c", "HOME": "/h"}):
            env = M.test_env()
        for name in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
            self.assertNotIn(name, env)
        self.assertEqual((env["THINK_MUTATION_CHECK"], env["HOME"]), ("1", "/h"))


class NotebookCommandContract(unittest.TestCase):
    """The commands in the cells are parsed by the real argparse parsers, so a renamed flag breaks a test, not a Kaggle session."""

    KEYS = tuple(NB.NOTEBOOKS)
    API_REV = C.QWEN_FP8_REVISION

    def test_every_decide_think_command_parses(self) -> None:
        variants = {"kaggle": [{}, {"LIMIT": ""}, {"BATCH_SIZE": 1, "MAX_TOKENS": 32768, "TIME_BUDGET_H": 1.5}],
                    "api": [{}, {"LIMIT": ""}, {"REVISION": self.API_REV, "CONCURRENCY": 2}, {"STANDARD_ONLY": True},
                            {"EXTRA_BODY": '{"enable_thinking": true, "note": "it\'s"}'}]}
        for key in self.KEYS:
            for params in variants[key]:
                with self.subTest(key=key, **{k: str(v)[:20] for k, v in params.items()}):
                    _, cli = parsed_commands(key, **params)
                    self.assertEqual(sorted(cli), ["compare", "sample", "thinker"])

    def test_the_three_commands_hand_files_to_each_other(self) -> None:
        for key in self.KEYS:
            ns, cli = parsed_commands(key)
            self.assertEqual(cli["thinker"].pack, cli["compare"].pack, key)
            self.assertEqual(cli["thinker"].pack, cli["sample"].pack, key)
            self.assertEqual(cli["thinker"].pack, ns["PACK"], key)
            self.assertEqual(cli["thinker"].out, cli["compare"].thinker, key)
            self.assertEqual(cli["thinker"].out, ns["WORK"] + "/results.jsonl", key)
            self.assertEqual(cli["compare"].out, ns["WORK"] + "/report", key)
            self.assertEqual(cli["sample"].cmd, "verify", key)
            self.assertEqual(cli["thinker"].manifest, cli["sample"].manifest, key)
            self.assertEqual(cli["thinker"].manifest, NB.MANIFEST_PATH, key)
        self.assertTrue((ROOT / NB.MANIFEST_PATH).is_file())

    def test_the_notebooks_run_the_protocol_as_registered(self) -> None:
        for key in self.KEYS:
            _, cli = parsed_commands(key)
            th = cli["thinker"]
            self.assertEqual(th.thinking, "on", key)
            self.assertEqual(th.max_tokens, T.DEFAULT_MAX_TOKENS, f"{key}: a notebook budget that is not the registered one")
            self.assertEqual([k for k in C.THINKING_SAMPLING if getattr(th, k) is not None], [], f"{key}: sampling is the model card's")
            self.assertEqual((th.overwrite, th.no_preflight, th.no_reasoning, th.seed, th.reasoning_effort), (False, False, False, None, None), key)
            self.assertEqual((th.keep_reasoning, th.retry_failed, th.max_attempts), (False, False, C.DEFAULT_MAX_ATTEMPTS), key)
            self.assertEqual(cli["compare"].bootstrap, C.BOOTSTRAP_B, key)
            self.assertEqual((cli["compare"].seed, cli["compare"].min_delta), (C.SEED, C.GT_MIN_DELTA_PTS), key)
            self.assertFalse(cli["compare"].ignore_pack_mismatch, key)

    def test_the_first_run_is_a_probe_of_three_rows(self) -> None:
        for key in self.KEYS:
            ns, cli = parsed_commands(key)
            self.assertEqual((ns["LIMIT"], cli["thinker"].limit), ("3", 3), key)
            _, full = parsed_commands(key, LIMIT="")
            self.assertIsNone(full["thinker"].limit, key)

    def test_kaggle_runs_the_nf4_transformers_backend_with_the_pinned_model(self) -> None:
        ns, cli = parsed_commands("kaggle")
        th = cli["thinker"]
        self.assertEqual((th.backend, th.quant, th.dtype, th.model), ("transformers", "nf4", "float16", C.QWEN_MODEL))
        self.assertEqual(th.batch_size, ns["BATCH_SIZE"])
        self.assertEqual(T.make_config(th)["revision"], C.QWEN_REVISION, "the default is the pinned revision")
        self.assertEqual(th.time_budget_s, ns["TIME_BUDGET_H"] * 3600)
        self.assertLessEqual(ns["TIME_BUDGET_H"], 9, "a Kaggle session is 9 to 12 hours; the run must stop before it")
        code = "\n".join(NB.code_cells("kaggle"))
        self.assertIn('shutil.copy(RESUME_FROM, f"{WORK}/results.jsonl")', code)

    def test_api_runs_the_openai_backend_and_declares_what_it_was_told(self) -> None:
        ns, cli = parsed_commands("api")
        th = cli["thinker"]
        self.assertEqual((th.backend, th.model, th.revision, th.concurrency), ("openai", ns["MODEL"], None, ns["CONCURRENCY"]))
        self.assertIn(ns["MODEL"], C.PINNED_REVISIONS)
        _, cli = parsed_commands("api", REVISION=self.API_REV)
        self.assertEqual(T.make_config(cli["thinker"])["revision"], self.API_REV)
        _, cli = parsed_commands("api", STANDARD_ONLY=True, EXTRA_BODY='{"a": "b c"}')
        self.assertTrue(cli["thinker"].standard_only)
        self.assertEqual(json.loads(cli["thinker"].extra_body), {"a": "b c"})

    def test_the_key_never_reaches_a_command_line_or_the_screen(self) -> None:
        api = "\n".join(NB.code_cells("api"))
        for _, cmd in NB.shell_commands("api"):
            self.assertNotRegex(cmd, r"(?i)api[-_]?key|bearer|authorization|THINK_API_KEY|OPENAI_API_KEY|\bhf_", cmd)
        self.assertNotRegex(api, r"print\([^)]*(\bkey\b|environ|KEY_NAME)")
        self.assertIn('os.environ["THINK_API_KEY"] = key', api)
        self.assertIn("del key", api)
        self.assertEqual(NB.parameters("api")["KEY_NAME"], "THINK_API_KEY", "the thinker reads this variable first")
        for key in self.KEYS:
            self.assertNotRegex("\n".join(NB.code_cells(key)), r"(?m)^\s*!\s*(env|printenv|set|export)\b", key)

    def test_the_local_vllm_cell_is_built_from_the_serve_module(self) -> None:
        api = "\n".join(NB.code_cells("api"))
        for needle in ("from decide.think import common, serve", "serve.vllm_command(MODEL, revision)", "serve.base_url()",
                       "serve.wait_ready(BASE_URL, 1800", 'REV = f"--revision {q(revision)}" if revision else ""', "server.terminate()"):
            self.assertIn(needle, api)
        self.assertEqual(NB.parameters("api")["MODE"], "endpoint", "a hosted endpoint is the default, not a 30 GiB download")

    def test_the_text_states_the_numbers_of_the_protocol(self) -> None:
        manifest = json.loads((ROOT / NB.MANIFEST_PATH).read_text(encoding="utf-8"))
        rows, benches = manifest["pack"]["rows"], len(manifest["benchmarks"])
        both = [f"{rows} linhas", f"**+{int(C.GT_MIN_DELTA_PTS)} pontos**", f"IC {int(round(100 * C.CI_LEVEL))}%", "ADR 0004", "nunca entra em treino"]
        for key in self.KEYS:
            text = "\n".join(NB.markdown_cells(key))
            for needle in both:
                self.assertIn(needle, text, f"{key}: {needle}")
            self.assertRegex(text, r"(?i)privado", key)
            self.assertIn("UNVERIFIED", text, key)
        kaggle = "\n".join(NB.markdown_cells("kaggle"))
        self.assertIn(f"{benches} provas", kaggle)
        self.assertIn(f"{C.PER_BENCHMARK} por prova", kaggle)

    def test_kaggle_says_what_happens_to_the_files_it_writes_in_working(self) -> None:
        """/kaggle/working is the output of the saved version: results.jsonl goes with it, and a public notebook has a public output (review of #78)."""
        text = "\n".join(NB.markdown_cells("kaggle"))
        self.assertIn("/kaggle/working", text)
        self.assertRegex(text, r"(?i)saída da versão salva|saida da versao salva")
        self.assertRegex(text, r"(?i)privado")
        self.assertEqual(NB.parameters("kaggle")["WORK"].split("/")[1:3], ["kaggle", "working"], "the default keeps the resume path that Save Version gives")

    def test_the_notebooks_say_what_each_verdict_status_means(self) -> None:
        for key in self.KEYS:
            text = "\n".join(NB.markdown_cells(key))
            for status in ("PASS", "FAIL", "PARTIAL", "EXPLORATORY", "INCONCLUSIVE"):
                self.assertIn(status, text, f"{key}: {status}")
        api = "\n".join(NB.markdown_cells("api"))
        self.assertIn("revision_not_declared", api, "an endpoint run that does not say which checkpoint it serves is not the gate")
        self.assertIn("standard_only", api)

    def test_the_area_readme_cites_both_notebooks(self) -> None:
        readme = (ROOT / "decide" / "think" / "README.md").read_text(encoding="utf-8")
        for path in NB.NOTEBOOKS.values():
            self.assertIn(str(path.relative_to(ROOT)), readme)


# ------------------------------------------------------------------------------------------- the README

README = ROOT / "decide" / "think" / "README.md"
README_PARSERS = {**CLI_PARSERS, "decide.think.hosted": H.build_parser, "decide.think.serve": SV.build_parser}


def plain(text):
    """Lower case, no accents: how the code spells a heading it points at."""
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch)).casefold()


def readme_text():
    return README.read_text(encoding="utf-8")


def readme_blocks(text):
    return re.findall(r"```[^\n]*\n(.*?)```", text, re.S)


def readme_lines(text):
    """Every command line of every fenced block, backslash continuations joined."""
    out = []
    for block in readme_blocks(text):
        buf = ""
        for line in block.splitlines():
            if line.rstrip().endswith("\\"):
                buf += line.rstrip()[:-1] + " "
                continue
            out.append((buf + line).strip())
            buf = ""
    return out


def modal_main_params():
    """(all parameters, required ones) of `main` in modal_app.py, read from the source: importing it needs `modal`."""
    import ast
    tree = ast.parse((ROOT / "decide" / "think" / "modal_app.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    names = [a.arg for a in main.args.args]
    return names, names[:len(names) - len(main.args.defaults)]


class ReadmeContract(unittest.TestCase):
    """decide/think/README.md says what the code does: a pasted command that stopped parsing, a pin that drifted or a heading
    the code points at that was renamed breaks a test, not somebody's afternoon."""

    def test_the_two_notebooks_are_cited_by_path(self) -> None:
        for path in NB.NOTEBOOKS.values():
            self.assertIn(str(path.relative_to(ROOT)), readme_text())

    def test_every_notebook_under_notebooks_decide_is_cited_by_the_readme_of_some_folder(self) -> None:
        """test_contract_30 only reads the first level of notebooks/ (a notebook of decide/ escapes it): this is the same rule for notebooks/decide/."""
        found = sorted((ROOT / "notebooks" / "decide").glob("*.ipynb"))
        self.assertTrue(found)
        readmes = [p for p in ROOT.rglob("README.md") if not {".git", "node_modules", ".venv", "venv"} & set(p.relative_to(ROOT).parts)]
        corpus = "\n".join(p.read_text(encoding="utf-8") for p in readmes)
        for nb in found:
            self.assertTrue(str(nb.relative_to(ROOT)) in corpus, f"{nb.name}: no README cites it")

    def test_every_status_and_reason_code_of_the_report_is_explained_in_the_readme(self) -> None:
        text = readme_text()
        for code in K.WHY:
            self.assertTrue(f"`{code}`" in text, f"the report prints {code}; the README does not say what it means")
        for status in ("PASS", "FAIL", "PARTIAL", "EXPLORATORY", "INCONCLUSIVE"):
            self.assertTrue(f"**{status}**" in text, status)
        for flag in ("--keep-reasoning", "--max-attempts", "--retry-failed", "--manifest"):
            self.assertTrue(flag in text, flag)

    def test_the_status_table_lists_the_reasons_of_each_status(self) -> None:
        rows = {m.group(1): m.group(2) for m in re.finditer(r"^  \| \*\*(INCONCLUSIVE|PARTIAL|EXPLORATORY)\*\* \| (.+) \|$", readme_text(), re.M)}
        self.assertEqual(sorted(rows), ["EXPLORATORY", "INCONCLUSIVE", "PARTIAL"])
        inconclusive = {"thinking_not_detected", "no_replies", "no_scored_rows", "no_decision_models"}
        partial = {"rows_not_run", "rows_with_transient_error"}
        exploratory = set(K.WHY) - inconclusive - partial
        for status, codes in (("INCONCLUSIVE", inconclusive), ("PARTIAL", partial), ("EXPLORATORY", exploratory)):
            listed = set(re.findall(r"`([a-z_]+)`", rows[status]))
            self.assertEqual(listed, codes, status)

    def test_the_readme_says_who_is_not_in_the_reference_and_what_the_key_and_the_working_folder_do(self) -> None:
        text = readme_text()
        quem = text[text.index("**Quem entra na referência.**"):]
        quem = quem[:quem.index("\n\n")]
        for name in ("mammoth-1", "pplx-decider-v1.2"):
            self.assertTrue(name in quem, name)
            self.assertTrue(name not in C.DECISION_MODELS, f"{name} has no per-row results: it cannot be in the reference")
        self.assertTrue("http://127.0.0.1:<porta>/v1" in text and "não segue redirecionamentos" in text)
        self.assertTrue("**Ela só viaja por `https://` ou para o próprio computador**" in text)
        self.assertTrue("`/kaggle/working` é a saída da versão salva" in text)
        for flag in ("--keep-reasoning", "--max-attempts", "--retry-failed"):
            self.assertTrue(text.count(flag) >= 1, flag)

    def test_the_tests_the_readme_names_exist(self) -> None:
        names = set(re.findall(r"`(test_[a-z0-9_]+)`", readme_text()))
        self.assertTrue(names)
        sources = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py"))
        for name in sorted(names):
            self.assertTrue(re.search(rf"def {name}\b", sources), f"the README names {name}, which is no test")

    def test_every_heading_the_code_points_at_exists(self) -> None:
        headings = {plain(h) for h in re.findall(r"^#{1,6}\s+(.+?)\s*$", readme_text(), re.M)}
        cited = set()
        for path in sorted((ROOT / "decide" / "think").glob("*.py")) + sorted((ROOT / "tests").glob("test_*_75.py")):
            for m in re.finditer(r'README, (?:\\?"([^"\\]+)\\?"|([A-Z][A-Za-z]+))', path.read_text(encoding="utf-8")):
                cited.add(plain(m.group(1) or m.group(2)))
        self.assertTrue({"backends", "tempo e custo", "se a saida for ruido"} <= cited, f"the scan found {sorted(cited)}")
        self.assertEqual(sorted(cited - headings), [], "a heading was renamed or removed")

    def test_the_relative_links_resolve(self) -> None:
        links = [m for m in re.findall(r"\]\(([^)\s]+)\)", readme_text()) if not m.startswith(("http://", "https://", "#"))]
        self.assertTrue(links)
        for link in links:
            self.assertTrue((README.parent / link.split("#")[0]).exists(), link)

    def test_there_is_no_link_to_main_that_the_layout_test_would_have_to_resolve(self) -> None:
        self.assertNotRegex(readme_text(), r"github\.com/simpletibr/simplicio-27b/(?:blob|tree)/main/")

    def test_every_python_command_parses_with_the_real_parser_of_its_module(self) -> None:
        seen = set()
        for line in readme_lines(readme_text()):
            m = re.match(r"python -m (decide\.think\.(\w+))(.*)$", line)
            if not m:
                continue
            module, name, rest = m.groups()
            self.assertTrue((ROOT / "decide" / "think" / f"{name}.py").is_file(), line)
            seen.add(module)
            if module in README_PARSERS:
                parse_cli(README_PARSERS[module](), shlex.split(rest))
        self.assertEqual(sorted(seen), ["decide.think.compare", "decide.think.hosted", "decide.think.mutation_check", "decide.think.sample",
                                        "decide.think.serve", "decide.think.thinker"])

    def test_the_modal_commands_use_the_arguments_of_the_entrypoint(self) -> None:
        names, required = modal_main_params()
        flags_ok = {"--" + n.replace("_", "-") for n in names}
        runs = [shlex.split(line) for line in readme_lines(readme_text()) if line.startswith("modal run ")]
        self.assertTrue(runs)
        for tokens in runs:
            self.assertEqual(tokens[:4], ["modal", "run", "-m", "decide.think.modal_app"], tokens)
            flags = {t for t in tokens[4:] if t.startswith("--")}
            self.assertLessEqual(flags, flags_ok, tokens)
            self.assertTrue({"--" + n.replace("_", "-") for n in required} <= flags, tokens)

    def test_the_pins_in_the_text_are_the_ones_in_the_code(self) -> None:
        text = readme_text()
        for model, rev in C.PINNED_REVISIONS.items():
            self.assertRegex(text, rf"`{re.escape(model)}` em `{rev}`")
        self.assertEqual(re.findall(r"vllm==[0-9][0-9.]*", text), [f"vllm=={SV.VLLM_VERSION}"])
        modal_pin = re.findall(r"modal==[0-9][0-9.]*", (ROOT / "decide" / "think" / "modal_app.py").read_text(encoding="utf-8"))
        self.assertEqual(sorted(set(re.findall(r"modal==[0-9][0-9.]*", text))), sorted(set(modal_pin)))
        self.assertIn(str(C.SEED), text)
        self.assertIn("16.384", text)
        self.assertEqual(T.DEFAULT_MAX_TOKENS, 16384)
        self.assertIn("+10,0", text)
        self.assertEqual(C.GT_MIN_DELTA_PTS, 10.0)
        self.assertIn("10.000 reamostras", text)
        self.assertEqual(C.BOOTSTRAP_B, 10_000)
        self.assertIn(f"max(32768, orçamento + {H.CONTEXT_MARGIN})", text)
        self.assertIn(f"**{H.EXIT_NO_SERVER}** o servidor não subiu", text)

    def test_the_pack_and_the_inputs_are_the_ones_of_the_manifest(self) -> None:
        manifest = json.loads(CommittedManifestContract.PATH.read_text(encoding="utf-8"))
        text = readme_text()
        self.assertIn(manifest["pack"]["content_sha256"], text)
        rows = re.findall(r"^\| (\S+) \| `([^`]+)` \| `([0-9a-f]{40})` \| `([^`]+)` \|$", text, re.M)
        self.assertEqual(sorted(r[0] for r in rows), manifest["decision_models"])
        self.assertEqual(len({r[2] for r in rows}), len(rows), "one revision per repository")
        sums = {}
        for digest, rel in re.findall(r"^([0-9a-f]{64})  (\S+)$", text, re.M):
            sums[rel.split("/")[0]] = (digest, rel.split("/")[1])
        self.assertEqual(sorted(sums), manifest["decision_models"])
        for model, repo, rev, path in rows:
            self.assertEqual(sums[model][0], manifest["inputs"]["results"][model]["sha256"], model)
            self.assertEqual(sums[model][1], path.rsplit("/", 1)[1], f"{model}: the file in the sha256 block is the one in the table")
        built = [shlex.split(line) for line in readme_lines(text) if line.startswith("python -m decide.think.sample build")]
        self.assertEqual(len(built), 1)
        named = sorted(a.split("=", 1)[0] for i, a in enumerate(built[0]) if i and built[0][i - 1] == "--results")
        self.assertEqual(named, manifest["decision_models"])
        for rel, info in manifest["inputs"]["suite_files"].items():
            self.assertIn(f"`{rel}`", text)

    def test_the_unverified_list_names_what_nobody_has_run(self) -> None:
        text = readme_text()
        tail = text[text.index("## O que não foi verificado"):]
        for word in ("GPU", "bitsandbytes", "vLLM", "Modal", "fp16", "latência", "SCHEMA.md"):
            self.assertIn(word, tail)
        self.assertIn("UNVERIFIED", text)

    def test_the_folder_ignores_the_private_files_by_name(self) -> None:
        patterns = (README.parent / ".gitignore").read_text(encoding="utf-8").splitlines()
        for pattern in ("*.jsonl", "*.jsonl.gz", "pack*/", "work/", "report/"):
            self.assertIn(pattern, patterns)
        self.assertFalse([p for p in patterns if p and not p.startswith("#") and "json" in p and not p.startswith("*.jsonl")],
                         "the manifest is tracked: no pattern may catch .json")

    def test_the_folder_keeps_the_private_files_out_of_git(self) -> None:
        import subprocess
        probe = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT, capture_output=True, text=True)
        if probe.returncode != 0:
            self.skipTest("not a git checkout")
        def ignored(name):
            return subprocess.run(["git", "check-ignore", "-q", f"decide/think/{name}"], cwd=ROOT).returncode == 0
        for private in ("pack.jsonl.gz", "results.jsonl", "pack-out/pack.jsonl.gz", "work/results.jsonl", "report/report.md"):
            self.assertTrue(ignored(private), private)
        for public in ("gt-sample-manifest.json", "README.md", "thinker.py"):
            self.assertFalse(ignored(public), public)


# ------------------------------------------------------------------------------------------- hosted: serve + think + compare

class FakeProc:
    """A `vllm serve` process that is up until it is terminated (or already ended, with `code`)."""

    def __init__(self, code=None, hangs=False):
        self.returncode, self.hangs, self.log = code, hangs, []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.log.append("terminate")
        if not self.hangs:
            self.returncode = 0

    def kill(self):
        self.log.append("kill")
        self.returncode = -9

    def wait(self, timeout=None):
        self.log.append("wait")
        if self.returncode is None:
            raise subprocess.TimeoutExpired("vllm", timeout)
        return self.returncode


class HostedContract(unittest.TestCase):
    def test_the_context_covers_the_prompt_and_the_whole_budget(self) -> None:
        self.assertEqual(H.context_length(16384), 32768)
        self.assertEqual(H.context_length(1), 32768)
        self.assertEqual(H.context_length(40000), 40000 + H.CONTEXT_MARGIN)

    def test_the_server_command_is_the_one_of_serve_py(self) -> None:
        from unittest import mock
        import tempfile
        with tempfile.TemporaryDirectory() as d, mock.patch.object(H.subprocess, "Popen") as popen:
            server = H.start_vllm(Path(d), C.QWEN_FP8_MODEL, C.QWEN_FP8_REVISION, port=8123, max_tokens=40000, concurrency=64,
                                  gpu_memory_utilization=0.8, tensor_parallel_size=2)
            cmd = popen.call_args[0][0]
            self.assertEqual(cmd, SV.vllm_command(C.QWEN_FP8_MODEL, C.QWEN_FP8_REVISION, port=8123, max_model_len=40000 + H.CONTEXT_MARGIN,
                                                  max_num_seqs=64, gpu_memory_utilization=0.8, tensor_parallel_size=2))
            self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.STDOUT)
            self.assertEqual(server.url, "http://127.0.0.1:8123/v1")
            self.assertTrue((Path(d) / "vllm.log").is_file())
            server.stop()
            self.assertIsNone(server.log_file)

    def test_few_requests_still_get_the_default_number_of_sequences(self) -> None:
        from unittest import mock
        import tempfile
        with tempfile.TemporaryDirectory() as d, mock.patch.object(H.subprocess, "Popen") as popen:
            H.start_vllm(Path(d), C.QWEN_FP8_MODEL, None, concurrency=4).stop()
            cmd = popen.call_args[0][0]
            self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "32")
            self.assertNotIn("--revision", cmd)

    def test_a_missing_vllm_binary_says_how_to_get_it(self) -> None:
        from unittest import mock
        import tempfile
        with tempfile.TemporaryDirectory() as d, mock.patch.object(H.subprocess, "Popen", side_effect=FileNotFoundError("vllm")):
            with self.assertRaises(FileNotFoundError) as cm:
                H.start_vllm(Path(d), C.QWEN_FP8_MODEL, None)
        self.assertIn(f"pip install vllm=={SV.VLLM_VERSION}", str(cm.exception))

    def test_stop_asks_nicely_then_kills_and_closes_the_log(self) -> None:
        import io
        quiet, stubborn, done = FakeProc(), FakeProc(hangs=True), FakeProc(code=0)
        for proc in (quiet, stubborn, done):
            log = io.StringIO()
            H.Server(proc, "u", log_file=log).stop(timeout_s=0.01)
            self.assertTrue(log.closed)
        self.assertEqual(quiet.log, ["terminate", "wait"])
        self.assertEqual(stubborn.log, ["terminate", "wait", "kill", "wait"])
        self.assertEqual(done.log, [])
        H.Server(None, "u").stop()                                    # an endpoint that was already up has no process

    def test_alive_follows_the_process(self) -> None:
        self.assertTrue(H.Server(None, "u").alive())
        self.assertTrue(H.Server(FakeProc(), "u").alive())
        self.assertFalse(H.Server(FakeProc(code=1), "u").alive())

    def test_tail_is_the_last_lines_of_the_log(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vllm.log"
            path.write_text("\n".join(f"line {i}" for i in range(50)), encoding="utf-8")
            self.assertEqual(H.Server(None, "u", path).tail(3), "line 47\nline 48\nline 49")
            self.assertEqual(H.Server(None, "u", Path(d) / "missing.log").tail(), "")
            self.assertEqual(H.Server(None, "u").tail(), "")

    @staticmethod
    def fake_clock():
        now = [0.0]
        return now, (lambda: now[0])

    def test_wait_returns_true_as_soon_as_the_server_answers(self) -> None:
        now, clock = self.fake_clock()
        seen = []

        def ready(url, timeout):
            seen.append((url, timeout))
            now[0] += timeout
            return len(seen) == 3
        self.assertTrue(H.wait_until_up(H.Server(FakeProc(), "http://h/v1"), 600, slice_s=15, ready=ready, clock=clock))
        self.assertEqual(len(seen), 3)
        self.assertEqual({u for u, _ in seen}, {"http://h/v1"})
        self.assertTrue(all(t <= 15 for _, t in seen))

    def test_wait_gives_up_when_the_process_ended_without_asking_the_endpoint(self) -> None:
        _, clock = self.fake_clock()

        def ready(url, timeout):
            raise AssertionError("the endpoint must not be asked once the process is gone")
        self.assertFalse(H.wait_until_up(H.Server(FakeProc(code=1), "u"), 600, ready=ready, clock=clock))

    def test_wait_gives_up_at_the_deadline_and_never_waits_past_it(self) -> None:
        now, clock = self.fake_clock()
        seen = []

        def ready(url, timeout):
            seen.append(timeout)
            now[0] += timeout
            return False
        self.assertFalse(H.wait_until_up(H.Server(None, "u"), 40, slice_s=15, ready=ready, clock=clock))
        self.assertEqual(seen, [15, 15, 10])

    def test_a_server_that_dies_midway_stops_the_wait(self) -> None:
        now, clock = self.fake_clock()
        proc, seen = FakeProc(), []

        def ready(url, timeout):
            seen.append(timeout)
            now[0] += timeout
            proc.returncode = 1 if len(seen) == 2 else None
            return False
        self.assertFalse(H.wait_until_up(H.Server(proc, "u"), 600, slice_s=15, ready=ready, clock=clock))
        self.assertEqual(len(seen), 2)

    def test_the_parser_defaults_are_the_registered_protocol(self) -> None:
        args = H.build_parser().parse_args(["--pack", "p", "--work", "w"])
        self.assertEqual((args.model, args.max_tokens, args.concurrency, args.port), (C.QWEN_FP8_MODEL, T.DEFAULT_MAX_TOKENS, 16, SV.DEFAULT_PORT))
        self.assertEqual(Path(args.manifest), ROOT / "decide" / "think" / "gt-sample-manifest.json")
        self.assertTrue(Path(args.manifest).is_file())
        self.assertIsNone(args.limit)

    def test_a_missing_pack_is_a_message_and_exit_2_not_a_traceback(self) -> None:
        import io
        import tempfile
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(err):
            self.assertEqual(H.main(["--pack", str(Path(d) / "nope.jsonl.gz"), "--work", d]), 2)
        self.assertIn("nope.jsonl.gz", err.getvalue())

    def test_the_exit_codes_are_the_documented_ones(self) -> None:
        self.assertEqual(H.EXIT_NO_SERVER, 4)
        for code in ("0 done", "2 the thinker or the pack refused", "3 the thinker", "4 the server did not come up"):
            self.assertIn(code, H.__doc__)

    def test_modal_pins_the_vllm_of_the_repository(self) -> None:
        text = (ROOT / "decide" / "think" / "modal_app.py").read_text(encoding="utf-8")
        self.assertIn(f'VLLM_PIN = "vllm=={SV.VLLM_VERSION}"', text)
        self.assertEqual(re.findall(r"modal==[0-9][0-9.]*", text), ["modal==1.6.2"], "one exact client version, the one this app was read against")
        self.assertEqual(re.findall(r"\bimport modal\b", (ROOT / "decide" / "think" / "hosted.py").read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()

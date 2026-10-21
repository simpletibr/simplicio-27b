"""Data pipeline of Simplicio-Decide-27B: corpus, decontamination, private proxy (issue #67, F2).

Formats are documented in SCHEMA.md and enforced by schema.py.  Everything here runs on CPU, streams its inputs
and resumes from the parts it already wrote.
"""

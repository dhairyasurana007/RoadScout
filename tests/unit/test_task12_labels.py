"""Task 12: load_labels (CSV label files) - parsing, validation and the optional-labels edge cases."""
from __future__ import annotations

import re

import pytest

from rs12_helpers import CLASSES, haz, seg, write_labels


def load(path):
    from backend.evaluation import load_labels

    return load_labels(path)


def total_rows(ls):
    return sum(len(d.segments) + len(d.hazards) for d in ls.drives.values())


# ------------------------------------------------------------------ optional labels

def test_missing_labels_dir_is_an_empty_label_set(tmp_path):
    ls = load(tmp_path / "does_not_exist")
    assert total_rows(ls) == 0
    assert ls.has_hazard_labels is False


def test_empty_labels_dir_is_an_empty_label_set(tmp_path):
    (tmp_path / "labels").mkdir()
    ls = load(tmp_path / "labels")
    assert total_rows(ls) == 0 and ls.has_hazard_labels is False


def test_str_and_path_arguments_are_both_accepted(tmp_path):
    write_labels(tmp_path / "labels", "IMG_1", [seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "pothole")])
    assert load(str(tmp_path / "labels")).has_hazard_labels is True
    assert load(tmp_path / "labels").has_hazard_labels is True


def test_other_files_and_subdirectories_are_ignored(tmp_path):
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg("A", "00:00", "01:00")])
    (d / "README.md").write_text("# not a label file", encoding="utf-8")
    (d / "notes.txt").write_text("x", encoding="utf-8")
    (d / "sub").mkdir()
    (d / "sub" / "IMG_2.csv").write_text("garbage", encoding="utf-8")
    ls = load(d)
    assert list(ls.drives) == ["img_1"]


def test_only_example_rows_gives_no_labels(tmp_path):
    d = tmp_path / "labels"
    write_labels(d, "IMG_6511", [
        seg("A", "00:00", "02:30", "EXAMPLE ROW: replace with a real 2-3 minute stretch"),
        haz("A", "01:12", "01:16", "debris in lane", "1", "H1", "EXAMPLE ROW: replace or delete"),
        seg("B", "10:00", "12:30", "EXAMPLE ROW: a stretch with nothing on the road (clean)"),
    ])
    ls = load(d)
    assert total_rows(ls) == 0
    assert ls.has_hazard_labels is False


def test_example_rows_are_ignored_even_if_they_would_be_invalid(tmp_path):
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [
        seg("A", "00:00", "02:30", "EXAMPLE ROW: segment"),
        haz("A", "09:00", "09:05", "not a real class", "9", "H1", "EXAMPLE ROW: outside, bad class and severity"),
        haz("Z", "xx:yy", "zz", "pothole", "1", "H2", "EXAMPLE ROW: unknown segment, bad time"),
    ])
    assert total_rows(load(d)) == 0


def test_example_rows_next_to_real_rows(tmp_path):
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [
        seg("A", "00:00", "02:30", "EXAMPLE ROW: replace"),
        haz("A", "01:12", "01:16", "debris in lane", "1", "H1", "EXAMPLE ROW: replace"),
        seg("R", "00:00", "05:00", "a real one"),
        haz("R", "01:00", "01:05", "pothole", "2", "H9", "real hazard"),
    ])
    ls = load(d)
    dl = ls.drives["img_1"]
    assert [s.segment_id for s in dl.segments] == ["R"]
    assert [h.hazard_id for h in dl.hazards] == ["H9"]
    assert ls.has_hazard_labels is True


def test_a_file_with_only_a_header_is_fine(tmp_path):
    write_labels(tmp_path / "labels", "IMG_1", [])
    ls = load(tmp_path / "labels")
    assert total_rows(ls) == 0 and ls.has_hazard_labels is False


def test_segments_without_hazards_do_not_count_as_hazard_labels(tmp_path):
    write_labels(tmp_path / "labels", "IMG_1", [seg("A", "00:00", "02:00")])
    ls = load(tmp_path / "labels")
    assert ls.has_hazard_labels is False
    assert [(s.segment_id, s.start, s.end) for s in ls.drives["img_1"].segments] == [("A", 0.0, 120.0)]


# ------------------------------------------------------------------ parsing

def test_fields_are_parsed(tmp_path):
    write_labels(tmp_path / "labels", "IMG_6513", [
        seg("A", "00:00", "02:30", "first stretch"),
        haz("A", "01:12", "01:16", "debris in lane", "1", "H1", "tipped barrel"),
        seg("B", "10:00", "1:02:03"),
        haz("B", "10:30", "10:31", "faded lane markings", "3", "", ""),
    ])
    ls = load(tmp_path / "labels")
    assert list(ls.drives) == ["img_6513"], "keyed by the lowercase file stem"
    dl = ls.drives["img_6513"]
    assert [(s.segment_id, s.start, s.end) for s in dl.segments] == [("A", 0.0, 150.0), ("B", 600.0, 3723.0)]
    h1, h2 = dl.hazards
    assert (h1.segment_id, h1.start, h1.end, h1.hazard_class, h1.severity, h1.hazard_id) == (
        "A", 72.0, 76.0, "debris in lane", 1, "H1")
    assert (h2.segment_id, h2.start, h2.end, h2.hazard_class, h2.severity) == (
        "B", 630.0, 631.0, "faded lane markings", 3)
    assert h2.hazard_id in (None, "")
    assert isinstance(h1.start, float) and isinstance(h1.severity, int)


def test_time_formats(tmp_path):
    write_labels(tmp_path / "labels", "IMG_1", [
        seg("A", "0:00", "2:05:09"), haz("A", "00:00", "00:59", "pothole"), haz("A", "1:00", "01:01", "pothole"),
        haz("A", "1:01:01", "1:01:02", "pothole")])
    hs = load(tmp_path / "labels").drives["img_1"].hazards
    assert [(h.start, h.end) for h in hs] == [(0.0, 59.0), (60.0, 61.0), (3661.0, 3662.0)]
    assert load(tmp_path / "labels").drives["img_1"].segments[0].end == 7509.0


@pytest.mark.parametrize("cls", CLASSES)
def test_every_valid_class_is_accepted(tmp_path, cls):
    write_labels(tmp_path / "labels", "IMG_1", [seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", cls)])
    assert load(tmp_path / "labels").drives["img_1"].hazards[0].hazard_class == cls


@pytest.mark.parametrize("sev", ["1", "2", "3"])
def test_valid_severities(tmp_path, sev):
    write_labels(tmp_path / "labels", "IMG_1", [seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "pothole", sev)])
    assert load(tmp_path / "labels").drives["img_1"].hazards[0].severity == int(sev)


def test_empty_severity_is_none(tmp_path):
    write_labels(tmp_path / "labels", "IMG_1", [seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "pothole", "")])
    assert load(tmp_path / "labels").drives["img_1"].hazards[0].severity is None


def test_hazard_exactly_on_the_segment_edges_is_inside(tmp_path):
    write_labels(tmp_path / "labels", "IMG_1", [seg("A", "00:10", "01:00"), haz("A", "00:10", "01:00", "pothole")])
    assert len(load(tmp_path / "labels").drives["img_1"].hazards) == 1


def test_crlf_bom_and_quoted_notes(tmp_path):
    p = write_labels(tmp_path / "labels", "IMG_1", [
        seg("A", "00:00", "01:00"), 'hazard,A,00:10,00:12,pothole,2,H1,"a note, with a comma"'],
        newline="\r\n", encoding="utf-8-sig")
    assert p.read_bytes().startswith(b"\xef\xbb\xbf")
    ls = load(tmp_path / "labels")
    assert ls.drives["img_1"].hazards[0].hazard_id == "H1"


def test_several_files_are_keyed_by_lowercase_stem(tmp_path):
    d = tmp_path / "labels"
    write_labels(d, "IMG_6511", [seg("A", "00:00", "01:00")])
    write_labels(d, "IMG_6513", [seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "pothole")])
    ls = load(d)
    assert sorted(ls.drives) == ["img_6511", "img_6513"]
    assert ls.has_hazard_labels is True


def test_the_repos_own_labels_dir_has_no_hazard_labels_yet():
    from rs12_helpers import ROOT

    labels = ROOT / "labels"
    if not labels.is_dir():
        pytest.skip("no labels dir")
    ls = load(labels)
    # whatever the user has labelled, loading must not raise; example-only files give no hazard labels
    assert isinstance(ls.has_hazard_labels, bool)


# ------------------------------------------------------------------ invalid rows

def where(msg, stem, line):
    return stem in msg and re.search(r"(?i)(line\s*%d\b|:%d\b)" % (line, line), msg) is not None


BAD = {
    "hazard_starts_before_its_segment": ([seg("A", "01:00", "02:00"), haz("A", "00:50", "01:10", "pothole")], 3),
    "hazard_ends_after_its_segment": ([seg("A", "01:00", "02:00"), haz("A", "01:50", "02:10", "pothole")], 3),
    "hazard_entirely_outside": ([seg("A", "00:00", "00:30"), haz("A", "05:00", "05:02", "pothole")], 3),
    "unknown_segment_id": ([seg("A", "00:00", "01:00"), haz("B", "00:10", "00:12", "pothole")], 3),
    "bad_time_text": ([seg("A", "00:00", "01:00"), haz("A", "abc", "00:12", "pothole")], 3),
    "bad_time_too_many_parts": ([seg("A", "00:00", "01:00"), haz("A", "00:10", "1:2:3:4", "pothole")], 3),
    "bad_time_empty": ([seg("A", "00:00", "01:00"), haz("A", "", "00:12", "pothole")], 3),
    "bad_segment_time": ([seg("A", "00:00", "xx:yy")], 2),
    "end_before_start": ([seg("A", "00:00", "01:00"), haz("A", "00:30", "00:20", "pothole")], 3),
    "segment_end_before_start": ([seg("A", "02:00", "01:00")], 2),
    "unknown_class": ([seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "sinkhole")], 3),
    "blank_class": ([seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "")], 3),
    "severity_zero": ([seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "pothole", "0")], 3),
    "severity_four": ([seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "pothole", "4")], 3),
    "severity_text": ([seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "pothole", "high")], 3),
    "unknown_row_type": ([seg("A", "00:00", "01:00"), "marker,A,00:10,00:12,pothole,1,,"], 3),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_invalid_rows_raise_a_value_error_naming_file_and_line(tmp_path, name):
    rows, line = BAD[name]
    write_labels(tmp_path / "labels", "IMG_9001", rows)
    with pytest.raises(ValueError) as ei:
        load(tmp_path / "labels")
    msg = str(ei.value)
    assert where(msg, "IMG_9001", line), "message must name the file and line %d: %s" % (line, msg)


def test_a_valid_row_after_an_example_row_keeps_the_right_line_number(tmp_path):
    rows = [seg("A", "00:00", "01:00", "EXAMPLE ROW"), seg("B", "00:00", "01:00"),
            haz("B", "00:10", "00:12", "sinkhole")]
    write_labels(tmp_path / "labels", "IMG_9002", rows)
    with pytest.raises(ValueError) as ei:
        load(tmp_path / "labels")
    assert where(str(ei.value), "IMG_9002", 4), str(ei.value)


def test_missing_header_column_is_an_error(tmp_path):
    write_labels(tmp_path / "labels", "IMG_9003", [seg("A", "00:00", "01:00")],
                 header="row_type,segment_id,start,end,class,severity,hazard_id")
    with pytest.raises(ValueError) as ei:
        load(tmp_path / "labels")
    assert "IMG_9003" in str(ei.value)


def test_a_bad_file_is_an_error_even_next_to_good_ones(tmp_path):
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "pothole")])
    write_labels(d, "IMG_2", [seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "nonsense")])
    with pytest.raises(ValueError) as ei:
        load(d)
    assert "IMG_2" in str(ei.value)

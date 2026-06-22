"""
Unit tests for the congress committee resolver (slice 4).

No network and no DB: the resolver is built from in-memory fixtures shaped
exactly like json.load() output for the three congress-legislators files.

Run from the TradingAgents repo root:
    python -m pytest tests/test_congress_committees.py -v
"""

from __future__ import annotations

import pytest

from tradingagents.dataflows.congress_committees import (
    NO_MATCH,
    CommitteeResolver,
    _normalize_name,
    enrich_committees,
)


# ─── Fixtures: shapes match the real congress-legislators JSON files ──────────

# committees-current.json is a list of committee dicts.
COMMITTEES = [
    {"type": "senate", "name": "Senate Committee on Armed Services", "thomas_id": "SSAS"},
    {"type": "senate", "name": "Senate Committee on Banking, Housing, and Urban Affairs",
     "thomas_id": "SSBK"},
    {"type": "senate", "name": "Senate Special Committee on Aging", "thomas_id": "SPAG"},
    {"type": "house", "name": "House Committee on Financial Services", "thomas_id": "HSBA"},
]

# committee-membership-current.json is a dict: committee/subcommittee id -> members.
# "SSAS01" is a SUBcommittee key and must be ignored (only top-level attributed).
MEMBERSHIPS = {
    "SSAS": [
        {"name": "Jack Reed", "bioguide": "R000122"},
        {"name": "Tommy Tuberville", "bioguide": "T000476"},
    ],
    "SSBK": [{"name": "Jack Reed", "bioguide": "R000122"}],
    "SPAG": [{"name": "Tommy Tuberville", "bioguide": "T000476"}],
    "SSAS01": [{"name": "Tommy Tuberville", "bioguide": "T000476"}],  # subcommittee — ignored
    "HSBA": [{"name": "French Hill", "bioguide": "H001072"}],
}

# legislators-current.json is a list; current party/chamber live in terms[-1].
LEGISLATORS = [
    {
        "id": {"bioguide": "R000122"},
        "name": {"first": "John", "last": "Reed", "official_full": "Jack Reed"},
        "terms": [
            {"type": "rep", "state": "RI", "party": "Democrat"},
            {"type": "sen", "state": "RI", "party": "Democrat"},
        ],
    },
    {
        "id": {"bioguide": "T000476"},
        "name": {"first": "Tommy", "last": "Tuberville", "official_full": "Tommy Tuberville"},
        "terms": [{"type": "sen", "state": "AL", "party": "Republican"}],
    },
    {
        "id": {"bioguide": "H001072"},
        "name": {"first": "French", "last": "Hill", "official_full": "J. French Hill"},
        "terms": [{"type": "rep", "state": "AR", "party": "Republican"}],
    },
    {
        # Same surname as a senator — proves chamber disambiguation and that an
        # ambiguous last-name-only lookup does not guess.
        "id": {"bioguide": "R000999"},
        "name": {"first": "Rhonda", "last": "Reed", "official_full": "Rhonda Reed"},
        "terms": [{"type": "rep", "state": "TX", "party": "Republican"}],
    },
]


@pytest.fixture
def resolver():
    return CommitteeResolver(
        legislators=LEGISLATORS, memberships=MEMBERSHIPS, committees=COMMITTEES
    )


# ─── Name normalization ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestNormalizeName:
    def test_simple(self):
        assert _normalize_name("Tommy Tuberville") == ("tommy tuberville", "tommy", "tuberville")

    def test_strips_title_and_suffix(self):
        assert _normalize_name("Sen. John Reed Jr.") == ("john reed", "john", "reed")

    def test_comma_order(self):
        assert _normalize_name("Tuberville, Tommy") == ("tommy tuberville", "tommy", "tuberville")

    def test_punctuation_and_middle_initial(self):
        full, first, last = _normalize_name("J. French Hill")
        assert first == "j" and last == "hill" and "french" in full

    def test_empty(self):
        assert _normalize_name("") == ("", "", "")
        assert _normalize_name(None) == ("", "", "")


# ─── Known members (committee populated) ──────────────────────────────────────


@pytest.mark.unit
class TestKnownMembers:
    def test_known_member_full_name(self, resolver):
        """AC: resolver maps a known member + chamber to the correct committee(s)."""
        m = resolver.resolve("Jack Reed", "Senate")
        assert m.matched
        assert m.bioguide == "R000122"
        assert m.party == "Democrat"
        assert m.chamber == "Senate"
        # Top-level committees only, short names, sorted; subcommittee excluded.
        assert m.committees == ("Armed Services", "Banking, Housing, and Urban Affairs")

    def test_committee_string_joins_multiple(self, resolver):
        m = resolver.resolve("Jack Reed", "Senate")
        assert m.committee_string == "Armed Services; Banking, Housing, and Urban Affairs"

    def test_committee_for_convenience(self, resolver):
        # "Senate Special Committee on Aging" shortens to "Aging".
        assert resolver.committee_for("Tommy Tuberville", "senate") == "Aging; Armed Services"

    def test_subcommittee_membership_not_attributed(self, resolver):
        # SSAS (top-level) + SPAG (top-level) count; SSAS01 (sub) does not.
        m = resolver.resolve("Tommy Tuberville", "senate")
        assert m.committees == ("Aging", "Armed Services")

    def test_comma_ordered_name(self, resolver):
        m = resolver.resolve("Tuberville, Tommy", "Senate")
        assert m.matched and m.bioguide == "T000476"

    def test_chamber_optional_when_unambiguous(self, resolver):
        m = resolver.resolve("Tommy Tuberville")
        assert m.matched and m.party == "Republican"

    def test_official_full_and_first_last_both_index(self, resolver):
        assert resolver.resolve("John Reed", "Senate").bioguide == "R000122"
        assert resolver.resolve("Jack Reed", "Senate").bioguide == "R000122"

    def test_house_member(self, resolver):
        m = resolver.resolve("French Hill", "House")
        assert m.matched and m.chamber == "House"
        assert m.committees == ("Financial Services",)


# ─── Unknown / ambiguous (committee null, no error) ───────────────────────────


@pytest.mark.unit
class TestUnknownMembers:
    def test_unknown_member_returns_no_match(self, resolver):
        """AC: unknown/unmatched members leave committee null without error."""
        m = resolver.resolve("Nobody McNobody", "Senate")
        assert not m.matched
        assert m.committees == ()
        assert m.committee_string is None
        assert m.party is None and m.chamber is None

    def test_committee_for_unknown_is_none(self, resolver):
        assert resolver.committee_for("Nobody McNobody", "House") is None

    def test_empty_name_is_no_match(self, resolver):
        assert not resolver.resolve("", "Senate").matched
        assert not resolver.resolve(None, "Senate").matched

    def test_ambiguous_last_name_does_not_guess(self, resolver):
        # House "Reed" (Rhonda) + Senate "Reed" (Jack): last-name-only, no chamber.
        assert not resolver.resolve("Reed").matched

    def test_chamber_disambiguates_shared_surname(self, resolver):
        m = resolver.resolve("Reed", "House")
        assert m.matched and m.bioguide == "R000999"

    def test_wrong_chamber_falls_back_not_crashes(self, resolver):
        # Senator queried as House: chamber filter empties but the full-name
        # lookup is still unique, so it resolves — and does not raise.
        m = resolver.resolve("Tommy Tuberville", "house")
        assert m.matched and m.bioguide == "T000476"

    def test_unknown_chamber_string_is_ignored(self, resolver):
        m = resolver.resolve("Tommy Tuberville", "Galactic Senate")
        assert m.matched and m.bioguide == "T000476"


# ─── Enrichment helper (used by the Senate/House ingest functions) ────────────


@pytest.mark.unit
class TestEnrichCommittees:
    def test_populates_committee_when_known(self, resolver):
        """AC: ingested rows have committee populated when the member is known."""
        rows = [
            {"member_name": "Jack Reed", "chamber": "Senate", "committee": None},
            {"member_name": "French Hill", "chamber": "House", "committee": None},
        ]
        enrich_committees(rows, resolver)
        assert rows[0]["committee"] == "Armed Services; Banking, Housing, and Urban Affairs"
        assert rows[1]["committee"] == "Financial Services"

    def test_unknown_member_left_null(self, resolver):
        rows = [{"member_name": "Nobody McNobody", "chamber": "Senate", "committee": None}]
        enrich_committees(rows, resolver)
        assert rows[0]["committee"] is None

    def test_none_resolver_is_noop(self):
        rows = [{"member_name": "Jack Reed", "chamber": "Senate", "committee": None}]
        enrich_committees(rows, None)
        assert rows[0]["committee"] is None


# ─── Robustness: never raises, degrades to no-match ───────────────────────────


@pytest.mark.unit
class TestRobustness:
    def test_empty_dataset_resolver(self):
        r = CommitteeResolver(legislators=[], memberships={}, committees=[])
        assert not r.resolve("Jack Reed", "Senate").matched
        assert r.committee_for("Jack Reed", "Senate") is None

    def test_malformed_legislator_records_are_skipped(self):
        legislators = [
            {"id": {}, "name": {}},                  # no bioguide
            {"id": {"bioguide": "X"}, "terms": []},  # no name, no terms
            None,                                    # junk
        ] + LEGISLATORS
        r = CommitteeResolver(legislators=legislators, memberships=MEMBERSHIPS,
                              committees=COMMITTEES)
        assert r.resolve("Tommy Tuberville", "senate").matched

    def test_member_with_no_committee_assignments(self):
        legislators = [{
            "id": {"bioguide": "Z000001"},
            "name": {"first": "Solo", "last": "Backbencher", "official_full": "Solo Backbencher"},
            "terms": [{"type": "rep", "party": "Independent"}],
        }]
        r = CommitteeResolver(legislators=legislators, memberships={}, committees=[])
        m = r.resolve("Solo Backbencher", "House")
        assert m.matched and m.committees == () and m.committee_string is None
        assert m.party == "Independent"

    def test_no_match_singleton_shape(self):
        assert NO_MATCH.committee_string is None and not NO_MATCH.matched

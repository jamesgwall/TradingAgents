"""
Committee enrichment for congressional-trades ingestion (slice 4).

Maps a disclosed trade's member + chamber to that member's standing-committee
assignments using the free, maintained ``unitedstates/congress-legislators``
dataset, which also supplies party + chamber normalization for free.

This is **best-effort enrichment**: the Senate/House ingest functions call the
resolver while normalizing rows and leave the ``committee`` column null when the
member can't be matched. Resolution never raises — an unknown member, a
malformed name, or an unreachable dataset all yield a no-match result, so
committee enrichment can never break ingestion.

Dataset (three published JSON files — stdlib ``json``, no extra dependency):
    legislators-current.json            roster: names, party, chamber (terms)
    committee-membership-current.json   committee/subcommittee id -> members
    committees-current.json             committee id -> human-readable name

CLI (demo / cache warm):
    python -m tradingagents.dataflows.congress_committees --name "Tommy Tuberville" --chamber senate
    python -m tradingagents.dataflows.congress_committees --refresh
    python -m tradingagents.dataflows.congress_committees --preflight

Env vars (all optional — defaults shown):
    CONGRESS_LEGISLATORS_BASE_URL      https://unitedstates.github.io/congress-legislators/
    CONGRESS_LEGISLATORS_CACHE_DIR     ~/.cache/tradingagents/congress-legislators
    CONGRESS_LEGISLATORS_MAX_AGE_DAYS  30
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ─── Config ─────────────────────────────────────────────────────────────────

DATASET_FILES = (
    "legislators-current.json",
    "committee-membership-current.json",
    "committees-current.json",
)

_BASE_URL = os.environ.get(
    "CONGRESS_LEGISLATORS_BASE_URL",
    "https://unitedstates.github.io/congress-legislators/",
)
_CACHE_DIR = Path(
    os.environ.get("CONGRESS_LEGISLATORS_CACHE_DIR")
    or (Path.home() / ".cache" / "tradingagents" / "congress-legislators")
)
_MAX_AGE_DAYS = int(os.environ.get("CONGRESS_LEGISLATORS_MAX_AGE_DAYS", "30"))
_HTTP_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_HTTP_TIMEOUT = 30

# Committee-name shortening, applied left-to-right to keep the stored value
# tight: drop "United States ", then an optional chamber qualifier, then the
# committee-type descriptor. Examples:
#   "Senate Committee on Armed Services"               -> "Armed Services"
#   "Senate Special Committee on Aging"                -> "Aging"
#   "House Permanent Select Committee on Intelligence" -> "Intelligence"
_CHAMBER_QUALIFIERS = ("Senate ", "House ", "Joint ")
_COMMITTEE_DESCRIPTORS = (
    "Permanent Select Committee on the ",
    "Permanent Select Committee on ",
    "Select Committee on the ",
    "Select Committee on ",
    "Special Committee on the ",
    "Special Committee on ",
    "Committee on the ",
    "Committee on ",
)

# Honorifics / titles dropped from a member name before matching.
_TITLES = frozenset(
    {"sen", "senator", "rep", "representative", "congressman", "congresswoman",
     "hon", "honorable", "the", "dr", "mr", "mrs", "ms", "del", "delegate"}
)
# Generational suffixes dropped before matching.
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# Incoming chamber strings -> dataset term ``type``.
_CHAMBER_TO_TYPE = {
    "senate": "sen", "sen": "sen", "s": "sen", "upper": "sen",
    "house": "rep", "rep": "rep", "representatives": "rep",
    "house of representatives": "rep", "h": "rep", "lower": "rep",
}
_TYPE_TO_CHAMBER = {"sen": "Senate", "rep": "House"}


# ─── Result type ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MemberMatch:
    """Result of resolving a (member_name, chamber) pair.

    ``matched`` is False for an unknown / ambiguous member; in that case
    ``committees`` is empty and the other fields are None.
    """
    matched: bool
    bioguide: str | None = None
    committees: tuple[str, ...] = ()
    party: str | None = None        # dataset value: "Democrat" / "Republican" / "Independent"
    chamber: str | None = None      # normalized "Senate" / "House"

    @property
    def committee_string(self) -> str | None:
        """Value for the ``committee`` text column: committees joined, or None."""
        return "; ".join(self.committees) if self.committees else None


NO_MATCH = MemberMatch(matched=False)


# ─── Name normalization ─────────────────────────────────────────────────────

def _norm_token(tok: str) -> str:
    return re.sub(r"[^a-z]", "", tok.lower())


def _normalize_name(raw: str | None) -> tuple[str, str, str]:
    """Return (full, first, last) normalized forms for matching.

    Handles "Last, First [Middle]" comma order, honorifics, generational
    suffixes, middle names/initials, and punctuation.
    """
    if not raw:
        return "", "", ""
    s = str(raw).strip()
    if "," in s:                                  # "Tuberville, Tommy" -> "Tommy Tuberville"
        last_part, _, rest = s.partition(",")
        s = f"{rest.strip()} {last_part.strip()}"

    tokens = [_norm_token(t) for t in s.split()]
    tokens = [t for t in tokens if t and t not in _TITLES and t not in _SUFFIXES]
    if not tokens:
        return "", "", ""
    return " ".join(tokens), tokens[0], tokens[-1]


# ─── Resolver ───────────────────────────────────────────────────────────────

class CommitteeResolver:
    """Resolve member + chamber -> committees from congress-legislators data.

    Construct from the three parsed dataset structures (the shapes produced by
    ``json.loads`` on each file — a list, a dict, and a list respectively), or
    via ``from_dataset_dir`` / ``load``.
    """

    def __init__(self, legislators: list, memberships: dict, committees: list):
        self._bioguide_to_committees: dict[str, tuple[str, ...]] = {}
        self._bioguide_meta: dict[str, dict] = {}
        self._by_full: dict[str, set[str]] = {}
        self._by_first_last: dict[tuple[str, str], set[str]] = {}
        self._by_last: dict[str, set[str]] = {}

        self._build_committees(committees, memberships)
        self._build_legislators(legislators)

    # ── index construction ──

    def _build_committees(self, committees: list, memberships: dict) -> None:
        """bioguide -> committee names, using only TOP-LEVEL committees.

        Membership keys are committee thomas_ids (e.g. "SSAS") and subcommittee
        keys (parent id + subcommittee id, e.g. "SSAS01"). Only top-level
        committees are attributed, so we key off the thomas_id set from
        committees-current.json.
        """
        id_to_name: dict[str, str] = {}
        for c in committees or []:
            tid = (c or {}).get("thomas_id")
            name = (c or {}).get("name")
            if tid and name:
                id_to_name[tid] = self._short_name(name)

        acc: dict[str, set[str]] = {}
        for committee_id, members in (memberships or {}).items():
            name = id_to_name.get(committee_id)
            if not name:                          # skip subcommittees / unknown ids
                continue
            for m in members or []:
                bioguide = (m or {}).get("bioguide")
                if bioguide:
                    acc.setdefault(bioguide, set()).add(name)
        self._bioguide_to_committees = {b: tuple(sorted(n)) for b, n in acc.items()}

    def _build_legislators(self, legislators: list) -> None:
        for leg in legislators or []:
            try:
                bioguide = (leg.get("id") or {}).get("bioguide")
                if not bioguide:
                    continue
                terms = leg.get("terms") or []
                current = terms[-1] if terms else {}
                self._bioguide_meta[bioguide] = {
                    "party": current.get("party"),
                    "type": (current.get("type") or "").lower(),   # "sen" / "rep"
                }

                name = leg.get("name") or {}
                official_full = name.get("official_full") or " ".join(
                    p for p in (name.get("first"), name.get("last")) if p
                )
                keys: list[str] = []
                if official_full:
                    keys.append(official_full)
                first, last = name.get("first"), name.get("last")
                if first and last:
                    keys.append(f"{first} {last}")
                nickname = name.get("nickname")
                if nickname and last:
                    keys.append(f"{nickname} {last}")

                for key in keys:
                    full, kfirst, klast = _normalize_name(key)
                    if full:
                        self._by_full.setdefault(full, set()).add(bioguide)
                    if kfirst and klast:
                        self._by_first_last.setdefault((kfirst, klast), set()).add(bioguide)
                    if klast:
                        self._by_last.setdefault(klast, set()).add(bioguide)
            except Exception:                      # one bad record must not break the index
                continue

    @staticmethod
    def _short_name(name: str) -> str:
        n = name
        if n.startswith("United States "):
            n = n[len("United States "):]
        for ch in _CHAMBER_QUALIFIERS:
            if n.startswith(ch):
                n = n[len(ch):]
                break
        for desc in _COMMITTEE_DESCRIPTORS:
            if n.startswith(desc):
                n = n[len(desc):]
                break
        return n

    # ── resolution ──

    def resolve(self, member_name: str, chamber: str | None = None) -> MemberMatch:
        """Best-effort match. Never raises; returns NO_MATCH when unsure."""
        try:
            return self._resolve(member_name, chamber)
        except Exception as e:                     # enrichment must never break ingestion
            log.debug("committee resolve failed for %r/%r: %s", member_name, chamber, e)
            return NO_MATCH

    def _resolve(self, member_name: str, chamber: str | None) -> MemberMatch:
        full, first, last = _normalize_name(member_name)
        if not last:
            return NO_MATCH
        want_type = _CHAMBER_TO_TYPE.get((chamber or "").strip().lower()) if chamber else None

        for candidates in (
            self._by_full.get(full, set()),
            self._by_first_last.get((first, last), set()) if first else set(),
            self._by_last.get(last, set()),
        ):
            picked = self._disambiguate(candidates, want_type)
            if picked:
                return self._match_for(picked)
        return NO_MATCH

    def _disambiguate(self, candidates: set[str], want_type: str | None) -> str | None:
        """Single bioguide if unambiguous after chamber filtering, else None."""
        if not candidates:
            return None
        if want_type:
            filtered = {b for b in candidates
                        if self._bioguide_meta.get(b, {}).get("type") == want_type}
            candidates = filtered or candidates
        return next(iter(candidates)) if len(candidates) == 1 else None

    def _match_for(self, bioguide: str) -> MemberMatch:
        meta = self._bioguide_meta.get(bioguide, {})
        return MemberMatch(
            matched=True,
            bioguide=bioguide,
            committees=self._bioguide_to_committees.get(bioguide, ()),
            party=meta.get("party"),
            chamber=_TYPE_TO_CHAMBER.get(meta.get("type", "")),
        )

    def committee_for(self, member_name: str, chamber: str | None = None) -> str | None:
        """Convenience: the value for the ``committee`` column (str), or None."""
        return self.resolve(member_name, chamber).committee_string

    # ── loading ──

    @classmethod
    def from_dataset_dir(cls, path) -> "CommitteeResolver":
        path = Path(path)
        loaded = {}
        for fname in DATASET_FILES:
            with open(path / fname, "r", encoding="utf-8") as fh:
                loaded[fname] = json.load(fh)
        return cls(
            legislators=loaded["legislators-current.json"],
            memberships=loaded["committee-membership-current.json"],
            committees=loaded["committees-current.json"],
        )

    @classmethod
    def load(
        cls,
        *,
        cache_dir=None,
        base_url: str | None = None,
        max_age_days: int | None = None,
        refresh: bool = False,
    ) -> "CommitteeResolver":
        """Load the resolver, refreshing the on-disk cache when stale/missing.

        Fails soft: if the dataset can't be fetched and no cache exists, returns
        an empty resolver (every lookup is a no-match) so committee enrichment
        degrades to null rather than breaking the caller.
        """
        cache_dir = Path(cache_dir or _CACHE_DIR)
        base_url = base_url or _BASE_URL
        max_age = _MAX_AGE_DAYS if max_age_days is None else max_age_days
        try:
            _refresh_cache(cache_dir, base_url, max_age, force=refresh)
            return cls.from_dataset_dir(cache_dir)
        except Exception as e:
            log.warning(
                "congress-legislators dataset unavailable (%s) — committee "
                "enrichment disabled for this run (committee left null)", e
            )
            return cls(legislators=[], memberships={}, committees=[])


# ─── Enrichment helper (used by the Senate/House ingest functions) ────────────

def enrich_committees(rows: list[dict], resolver: "CommitteeResolver | None") -> list[dict]:
    """Populate each row's ``committee`` from the resolver, in place.

    Best-effort: a None resolver is a no-op, and any member that doesn't resolve
    keeps ``committee`` as-is (null). Returns the same list for convenience.
    """
    if resolver is None:
        return rows
    for row in rows:
        committee = resolver.committee_for(row.get("member_name"), row.get("chamber"))
        if committee:
            row["committee"] = committee
    return rows


# ─── Cache / download ─────────────────────────────────────────────────────────

def _is_stale(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return True
    if max_age_days <= 0:
        return False
    return (time.time() - path.stat().st_mtime) / 86400.0 > max_age_days


def _refresh_cache(cache_dir: Path, base_url: str, max_age_days: int, *, force: bool) -> None:
    """Ensure all dataset files are present and fresh in ``cache_dir``.

    Downloads to a temp file and atomically renames, so a partial download can't
    leave a corrupt cache file. A stale-but-present file is kept if re-download
    fails.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fname in DATASET_FILES:
        dest = cache_dir / fname
        if not force and not _is_stale(dest, max_age_days):
            continue
        url = base_url.rstrip("/") + "/" + fname
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _HTTP_UA})
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                data = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
            log.info("congress-legislators: refreshed %s (%d bytes)", fname, len(data))
        except Exception as e:
            if dest.exists():
                log.warning("congress-legislators: re-download of %s failed (%s) — "
                            "using existing cached copy", fname, e)
            else:
                raise


# ─── Module-level default resolver (lazy singleton) ───────────────────────────

_default_resolver: "CommitteeResolver | None" = None


def get_default_resolver(*, refresh: bool = False) -> CommitteeResolver:
    """Lazily load and memoize a process-wide resolver for the ingestor."""
    global _default_resolver
    if _default_resolver is None or refresh:
        _default_resolver = CommitteeResolver.load(refresh=refresh)
    return _default_resolver


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Congress committee resolver (slice 4)")
    parser.add_argument("--name", help="member name to resolve")
    parser.add_argument("--chamber", help="senate | house (optional disambiguator)")
    parser.add_argument("--refresh", action="store_true", help="force re-download of the dataset")
    parser.add_argument("--preflight", action="store_true", help="load and report index sizes")
    parser.add_argument("-v", "--verbose", action="store_true", help="log INFO progress")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if (args.verbose or args.preflight) else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    resolver = CommitteeResolver.load(refresh=args.refresh)

    if args.preflight or not args.name:
        log.info("Loaded %d current legislators; %d have committee assignments.",
                 len(resolver._bioguide_meta), len(resolver._bioguide_to_committees))
        if not args.name:
            return 0

    match = resolver.resolve(args.name, args.chamber)
    if match.matched:
        print(json.dumps({
            "name": args.name, "chamber": match.chamber, "party": match.party,
            "committees": list(match.committees), "committee_string": match.committee_string,
        }, indent=2))
    else:
        print(json.dumps({"name": args.name, "chamber": args.chamber, "matched": False,
                          "committee": None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

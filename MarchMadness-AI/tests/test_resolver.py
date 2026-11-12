import pandas as pd
import pytest

from mmai.ids.resolver import PROVISIONAL_BASE, TeamResolver, UnresolvedTeamError, is_provisional, provisional_id


def _resolver():
    return TeamResolver(pd.DataFrame([
        {"gender": "M", "source": "espn", "key": "193", "kaggle_id": 1275},
        {"gender": "M", "source": "espn", "key": "2390", "kaggle_id": 1274},
        {"gender": "M", "source": "odds_api", "key": "Miami Hurricanes", "kaggle_id": 1274},
        {"gender": "W", "source": "espn", "key": "193", "kaggle_id": 3275},
    ]))


def test_resolves_by_gender_and_source():
    r = _resolver()
    assert r.resolve("M", "espn", 193) == 1275
    assert r.resolve("W", "espn", "193") == 3275
    assert r.resolve("M", "odds_api", "Miami Hurricanes") == 1274


def test_unknown_key_raises_instead_of_guessing():
    with pytest.raises(UnresolvedTeamError, match="team_aliases.csv"):
        _resolver().resolve("M", "odds_api", "Miami RedHawks")


def test_resolve_many_reports_missing():
    found, missing = _resolver().resolve_many("M", "espn", ["193", "2390", "193", "nope"])
    assert found == {"193": 1275, "2390": 1274} and missing == ["nope"]


def test_aliases_override_learned_crosswalk(tmp_path):
    xw = tmp_path / "xw.csv"
    al = tmp_path / "aliases.csv"
    pd.DataFrame([{"gender": "M", "source": "espn", "key": "193", "kaggle_id": 9999, "votes": 30}]).to_csv(xw, index=False)
    al.write_text("# comment line\ngender,source,key,kaggle_id,note\nM,espn,193,1275,checked game log\n")
    assert TeamResolver.load(xw, al).resolve("M", "espn", "193") == 1275


def test_load_without_files_gives_empty_resolver(tmp_path):
    r = TeamResolver.load(tmp_path / "missing.csv", tmp_path / "missing2.csv")
    with pytest.raises(UnresolvedTeamError):
        r.resolve("M", "espn", "1")


def test_provisional_ids_never_collide_with_kaggle_ids():
    pid = provisional_id("131234")
    assert pid > PROVISIONAL_BASE and is_provisional(pid) and not is_provisional(1181)

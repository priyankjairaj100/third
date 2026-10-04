# Event replication and chronological requests

`replication.py` completes original-schema WCEP event intake, whole-event panel
selection, and descriptive date-group withdrawals on already prepared News.
The 24 current checks use temporary schema/calendar fixtures only. No WCEP
archive was acquired, no News dates were invented, and no replication experiment
has been run.

The input contract was inspected against the author's source at immutable commit
`8aa8418d55a4e6710277bed18bfc5349b50a5d96`:

- https://github.com/complementizer/wcep-mds-dataset/tree/8aa8418d55a4e6710277bed18bfc5349b50a5d96
- `dataset_reproduction/combine_and_split.py`, upstream blob
  `0fc9306af2ee15b87a24d58967f407ad01da52cd`
- `dataset_reproduction/extract_wcep_articles.py`, upstream blob
  `b518e9d599c6e3e3d0971690462ee5f81453d979`
- `dataset_reproduction/extract_cc_articles.py`, upstream blob
  `9dfc1eb441063f54a5fd683a4bb0c2159c09bf10`

The original author's extracted format is event JSONL (possibly gzipped), with
native event `id`, `date`, `collection`, and an `articles` array. Articles have
their own `title`, `text`, and available original provenance. Processed releases
containing concatenated documents do not satisfy this contract. No article is
downloaded again or regenerated from current web pages. The repository software
license does not establish permission to redistribute publisher text.

## Intake and selection

```bash
python3 empirical_execution/phase5/replication.py wcep train.jsonl.gz val.jsonl.gz test.jsonl.gz \
  --out /absolute/local/wcep_parsed --snapshot-revision ACTUAL_RELEASE \
  --collection-scope all_official_splits
python3 empirical_execution/phase5/replication.py panel /absolute/local/wcep_parsed \
  --out /absolute/local/wcep_panel --target 25000
```

Original file SHA256, native event ID and article ordinal determine instance
identity. Repeated URLs, article IDs and text remain separate natural instances.
Duplicate event IDs across the declared inputs fail intake. Native collections
are preserved; declaring all splits requires all three. Empty texts receive
explicit exclusions. No fresh 100-article cap is applied: the declared original
WCEP-100 release must supply its own membership.

Title/body construction reuses the versioned News rule. Event summary, category,
reference links and other metadata are kept in `events.jsonl`, separate from
encoder inputs. Event dates are event metadata, not inferred article publication
times. Event membership is never used as a duplicate label.

The panel takes events in SHA256(`ccu-v1-wcep-event-panel` + NUL + native event ID)
order until the 25,000-article soft target is reached, including the entire
boundary event. Hash ties use native ID. Empty events, overshoot and shortfall
are visible. Completeness refers to surviving instances of the supplied events;
the parser cannot prove the supplied original snapshot is complete or authentic.
SQLite limits the resident native-ID index, but an entire JSONL event and its
articles are resident while parsing. Panel selection retains event metadata/IDs
in memory; it does not assert constant memory.

The default is the protocol-permitted **record-only WCEP branch**. URL strings
are preserved but are not promoted to verified source ownership or registrable
domains. A source-withdrawal branch requires independently verified original URL
and pinned PSL bindings. The primary and secondary WCEP curators reuse their
respective frozen CC-News thresholds; there is no WCEP retuning. WCEP is an
event-enriched replication, not a random sample of news or general web text.

## News chronological stream

```bash
python3 empirical_execution/phase5/replication.py chronological guarded_news.jsonl \
  --out /absolute/local/date_requests.json --horizon 128 --resolution day
```

Inputs must already be training-partition rows with original recorded dates in
2018–2019. Calibration/test and 2017 rows are refused. The ordered stream removes
complete calendar days (or a separately declared month resolution), stable-ID
ties within a date, with each release recording cumulative IDs and counts. A
group that exceeds the remaining locked record horizon stops the stream before
that group; it is neither split nor skipped to find a smaller later group. The
blocked group and remaining budget are recorded. Use a separately constructed
larger-horizon state if that policy was declared before the run.

This is one descriptive sequence of simulated requests, with zero independently
sampled trajectory replicates. Dates are the recorded calendar dates; timestamps
are not reinterpreted into an invented timezone. The sealed population binding
guards against accidental row changes but does not prove upstream authenticity
or semantic-guard completion.

No publisher article bodies from these workflows belong in the public backup.
Keep local input/output corpora outside tracked paths and release only permitted
IDs, hashes, code and derived results under the existing packaging policy.

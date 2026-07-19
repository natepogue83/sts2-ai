# Data provenance

Raw and processed datasets are intentionally ignored by version control. This
file records how to reproduce them without redistributing player data.

## STS2 Runs snapshot

- Listing: https://sts2runs.com/downloads
- Download API: https://sts2runs.com/api/downloads
- File: `runs-all-before-2026-06.json.gz`
- Retrieved: 2026-07-13
- Advertised run count: 6,796
- Size: 15,574,676 bytes
- SHA-256: `1C207D44F497D867C2B227093FE75541CDDAE538CCF3C68886316A15B3FF6F32`
- Format: gzip-compressed NDJSON; server-added cheat flags are preserved.

The site explicitly publishes this file as a community data download but no
separate dataset license was visible at retrieval time. Use it for this research
project, retain provenance, respect deletion/privacy requests, and do not
redistribute the raw snapshot without confirming permission.

## Local run history

- Auto-detected under `%APPDATA%\SlayTheSpire2`.
- Raw files remain outside the repository.
- Derived records contain an anonymized run digest but no player/Steam id.


# `reference-unequal-v1` compatibility evidence

This directory is the committable evidence for the preserved
`data/ls20-reference-unequal-v1` bank. `bank.py.txt` and `generate.py.txt` are
byte-for-byte copies of the generator sources named by the bank's historical
manifest. Their SHA256 values remain `dd5e7cf4...ee454` and
`90a7fe0e...a4c81`, respectively. The two JSON files retain the historical
generator receipt and its passed preservation verification.

`compatibility-receipt.json` binds the unchanged manifest digest, the archived
source digests, the reviewed current generator digests, and the seven stored
reference fixture replays. The fixture proof is a structural compatibility
gate; it is not a claim that all 10,500 stored rows were rebuilt byte for byte.

The evidence is specific to this bank and its fixed repository-relative paths.
Changing the manifest, either generator source, any listed source, this
evidence, or a fixture fails the read-only bank catalogue closed. The original
execution archive under `artifacts/spatial-repair-v4/` is retained separately
as historical provenance; runtime validation uses this committable directory.

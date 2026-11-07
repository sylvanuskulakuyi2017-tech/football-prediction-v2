# Data provenance

Source: Mart Jürisoo, international football results.
Repository: https://github.com/martj42/international_results
Upstream licence: CC0-1.0 (https://creativecommons.org/publicdomain/zero/1.0/).
Upstream licence text: https://github.com/martj42/international_results/blob/master/LICENSE

The original results.csv snapshot is retained byte-for-byte. Its SHA-256 is
`09aac809e317f3004c69720a38addb656c49ebb19f1937deab96f96e3362257a`. Preparation keeps matches from 1990-01-01 using
the original as-of cutoff 2026-11-07. It produces 32,821
matches with 327 recorded team names.

The snapshot's latest included match is 2026-10-06; this project does not
claim the source is current to the date a user opens the app. Historical results,
country names and competition coverage may contain errors or omissions.

Raw validation warnings remain in reports/dataset_validation.json. Duplicate or
conflicting fixture warnings must be interpreted with their dates and the chosen
modelling window. Preparation does not rewrite the raw file. The current prepared
window has 0 repeated fixture keys.

The MIT licence for project code does not replace the data's CC0 status or the
licences of dependencies. Optional upstream README/LICENSE files, when present,
are included with the raw snapshot.

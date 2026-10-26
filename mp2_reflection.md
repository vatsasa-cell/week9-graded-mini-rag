# MP2 Reflection

## What worked

The ingestion pipeline successfully loaded all five Sherlock Holmes story summaries, created 89 chunks, and stored them in the Qdrant collection `mp2_sherlock_gorthi`. All five validation questions retrieved their expected source files. For example, `q3_blue_carbuncle_goose` retrieved `03_blue_carbuncle.txt` and included both expected fact phrases in its answer. The source citations helped connect generated answers to the relevant story summaries.

## What didn't work

Source retrieval was successful, but answer completeness was inconsistent. For `q1_red_headed_league_assistant`, the expected source was retrieved, but only 1 of 5 expected fact phrases was present. The answer identified Vincent Spaulding but did not establish his real identity. For `q2_speckled_band_murder_weapon`, the expected source was also retrieved, but only 4 of 6 expected fact phrases were present. These results show that retrieving the correct document does not guarantee a complete answer. The validation's PASS status primarily reflects source matching, so it should not be interpreted as proof of full answer accuracy.

## What I'd change

With another five hours, I would investigate retrieval and answer generation for questions that require multiple facts or a character's hidden identity. I would inspect the relevant chunks, test different top-k values and chunk sizes, and check whether the necessary evidence reaches the language model. I would also improve the validation report so source matching and expected-fact coverage are reported as separate measures. This would make it easier to identify whether a failure comes from retrieval, incomplete context, or answer generation.

## One surprise

I was surprised that all five questions matched their expected source files while the first two predefined questions still had missing expected fact phrases. The learner-authored questions achieved full expected-fact coverage (2/2, 3/3, and 2/2), whereas the predefined questions achieved 1/5 and 4/6. This reinforced the importance of evaluating answer completeness separately from retrieval success.

## Validation summary

The actual validation run is recorded in `mp2_validation.txt`.

- Documents ingested: 5
- Chunks created and stored: 89
- Predefined questions: 2/2 expected source matches
- Learner-authored questions: 3/3 expected source matches
- Overall expected source matches: 5/5
- Expected-fact phrase coverage:
  - `q1_red_headed_league_assistant`: 1/5
  - `q2_speckled_band_murder_weapon`: 4/6
  - `q3_blue_carbuncle_goose`: 2/2
  - `q4_engineers_thumb_hatherley`: 3/3
  - `q5_scandal_irene_adler`: 2/2

The PASS labels indicate that the expected source files were retrieved. They do not establish that every expected fact was included or that every answer was fully accurate.

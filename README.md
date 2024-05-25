
# Index data
The sample review data in raw format is in 

  review/

First the reviews must be indexed. 
Run (avoid trailing slash!): 

  python3 index.py reviews


Press 0 to generate the lookups.
Run one time with both 0 and 2.

  echo "0" | python3 index.py  reviews;
  echo "2" | python3 index.py  reviews;


# Run analysis
To start the analysis, run: 

  python3 analyze.py

reviews_lookup is used for most methods, 
while reviews_multiLookup is an optimization used for CoR. 


## For ATW, pick:
reviews_lookup
1 : Aggregated time windows
60  (For 60 minutes, feel free to test others)

Result in: out/reviews_lookup/atw_60 

## The HVC method is packaged separately in hvc folder
Follow README in hvc folder

## For CoR, pick:
reviews_multiLookup
coAuthor

Result in: out/reviews_multiLookup/coAuthor

# For spam: 
reviews_lookup
11
3 (for 3 min)

Result in: out/reviews_lookup/spam_detection_3


# For Written Ratio
reviews_lookup
7

Result in: out/reviews_lookup/written_ratio



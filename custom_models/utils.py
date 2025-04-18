from rapidfuzz import process


def get_movie_by_id(df, movie_id):
    try:
        return df.loc[df["ITEM_ID"] == int(movie_id)][['ITEM_ID', 'TITLE', 'GENRES', 'TYPE']].values[0]
    except Exception as e:
        return "Error obtaining title"


def get_top_matched_titles(df, input_title, top_n=10, min_score=75):
    input_title = input_title.lower()
    results = []
    seen_ids = set()

    exact_match = df[df['TITLE'].str.lower() == input_title]
    if not exact_match.empty:
        for _, row in exact_match.iterrows():
            seen_ids.add(row['ITEM_ID'])
            results.append(row['ITEM_ID'])

    all_titles = df['TITLE'].str.lower().tolist()
    matches = process.extract(
        input_title,
        all_titles,
        limit=top_n * 2,
        score_cutoff=min_score
    )

    for matched_title_lower, score, _ in matches:
        if len(matched_title_lower) < 3:
            continue

        matched_rows = df[df['TITLE'].str.lower() == matched_title_lower]
        if not matched_rows.empty:
            item_id = matched_rows.iloc[0]['ITEM_ID']
            if item_id in seen_ids:
                continue

            results.append(item_id)
            seen_ids.add(item_id)

    return results[: top_n]


def get_item_id_from_title(df, input_title, min_score=90):
    all_titles = df['TITLE'].tolist()
    match = process.extractOne(input_title, all_titles, score_cutoff=min_score)

    if match:
        matched_title = match[0]
        item_id = df[df['TITLE'] == matched_title]['ITEM_ID'].values[0]
        return item_id
    else:
        return None

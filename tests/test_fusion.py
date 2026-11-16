from rag_agent.fusion import rrf_fuse


def test_rrf_merges_two_rankings():
    result = rrf_fuse([10, 11, 12], [12, 13], limit=10)
    ids = [item[0] for item in result]
    scores = dict(result)

    # 12 同时出现在两路，应该排在第一。
    assert ids[0] == 12
    assert scores[12] > scores[10]
    assert set(ids) == {10, 11, 12, 13}


def test_rrf_respects_limit():
    result = rrf_fuse([1, 2, 3], [3, 4, 5], limit=2)
    assert len(result) == 2


def test_rrf_empty():
    assert rrf_fuse([], [], limit=20) == []


def test_rrf_scores_sorted_desc():
    result = rrf_fuse([1, 2], [2, 3], limit=10)
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)

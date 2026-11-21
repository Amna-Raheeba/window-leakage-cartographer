from window_leakage.predictor import KNNRegressor


def test_knn_exact_match():
    model = KNNRegressor(k=1)
    model.fit([(0.0, 0.0), (10.0, 10.0)], [1.0, 2.0])
    assert model.predict_one((0.0, 0.0)) == 1.0
    assert model.predict_one((10.0, 10.0)) == 2.0


def test_knn_averages_k_neighbours():
    model = KNNRegressor(k=2)
    model.fit([(0.0,), (1.0,), (100.0,)], [0.0, 2.0, 50.0])
    assert model.predict_one((0.4,)) == 1.0


def test_knn_k_larger_than_train():
    model = KNNRegressor(k=10)
    model.fit([(0.0,), (2.0,)], [1.0, 3.0])
    assert model.predict_one((1.0,)) == 2.0


def test_knn_empty_train():
    model = KNNRegressor(k=3)
    model.fit([], [])
    assert model.predict_one((1.0,)) == 0.0


def test_knn_standardisation_scale_invariance():
    # A feature measured in thousands should not dominate one in units.
    model = KNNRegressor(k=1)
    model.fit([(0.0, 0.0), (1000.0, 1.0)], [0.0, 10.0])
    assert model.predict_one((900.0, 0.9)) == 10.0
    assert model.predict_one((100.0, 0.1)) == 0.0


def test_knn_predict_list():
    model = KNNRegressor(k=1)
    model.fit([(0.0,), (5.0,)], [1.0, 2.0])
    assert model.predict([(0.1,), (4.9,)]) == [1.0, 2.0]

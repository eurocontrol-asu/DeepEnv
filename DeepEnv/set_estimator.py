import warnings

import numpy as np
import pandas as pd
import pkg_resources
import tensorflow as tf

absolute_angle = lambda el: np.abs(el) if np.abs(el) < 180 else 360 - np.abs(el)


class SetEstimator:
    """
    Class that contains data pipelines for Single Engine Taxiing identification
    """

    def __init__(self, classifier_path: str = None, regressor_path: str = None, padding_size=2048):
        """
        Initializes the SetEstimator class.

        Args:
            classifier_path (str): The path to the classification model. Default is None (use package data).
            regressor_path (str): The path to the regression model. Default is None (use package data).

        """

        self.padding_size = padding_size

        if classifier_path is None:
            classifier_path = pkg_resources.resource_filename(
                "DeepEnv", "models/SET/SET_A320_V0.1/classifier"
            )

        if regressor_path is None:
            regressor_path = pkg_resources.resource_filename(
                "DeepEnv", "models/SET/SET_A320_V0.1/regressor"
            )

        classifier_model = tf.saved_model.load(classifier_path)
        self.classifier = classifier_model.signatures[
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]

        regressor_model = tf.saved_model.load(regressor_path)
        self.regressor = regressor_model.signatures[
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]

    def estimate(self, flight: pd.DataFrame, **kwargs) -> (float, int):
        """
            Return the probability of single engine taxiing and give an estimate of the index of start

            The minimum set of features are:
                - flight (pd.DataFrame): The flight data as a pandas DataFrame.
                - groundspeed (str): The column name for the groundspeed (in knot).
                    Default is "groundspeed".
                - altitude (str): The column name for the altitude (in feet).
                    Default is "altitude".
                - track_angle (str): The column name for the track angle (in °).
                    Default is "track_angle".
                - second (str): The column name for the timestamp (in second).
                    Default is "second".
                - on_taxiway (str): The column name for the on_taxiway (0, 1).
                    Default is "on_taxiway".

            Returns:
                (float, int) : A tuple of the probability of being a single engine taxi and the estimate start index.
              Note:

                - When the `second` column is provided, the set estimator is more accurate,
                    especially due to **derivatives of speeds and track angle** used in the model.
                - Expected sampling rate is 1 seconds, higher or lower sampling rate might induce errors. Resampling data before applying the set estimator is recommanded.

            For an example of use, refer to `examples/set_estimator/example.ipynb`

        """

        col_groundspeed = kwargs.get("groundspeed", "groundspeed")
        col_altitude = kwargs.get("altitude", "altitude")
        col_track_angle = kwargs.get("track_angle", "track_angle")
        col_on_taxiway = kwargs.get("on_taxiway", "on_taxiway")
        col_second = kwargs.get("second", "second")

        assert col_groundspeed in flight.columns, f"Column {col_groundspeed} not found"
        assert col_altitude in flight.columns, f"Column {col_altitude} not found"
        assert col_track_angle in flight.columns, f"Column {col_track_angle} not found"
        assert col_second in flight.columns, f"Column {col_second} not found"
        assert col_on_taxiway in flight.columns, f"Column {col_on_taxiway} not found"
        assert (flight[col_second].dtype == float or int), "column for second must be float or integer"

        flight = flight.assign(
            dt5=lambda d: d[col_second].diff(5).bfill(),
            dt10=lambda d: d[col_second].diff(10).bfill()
        ).assign(
            d5_groundspeed=lambda d: (d[col_groundspeed].diff(5).bfill() / d.dt5),
            d10_groundspeed=lambda d: (d[col_groundspeed].diff(10).bfill() / d.dt10),
            d5_track_angle=lambda d: (d[col_track_angle].diff(5).bfill().apply(absolute_angle) / d.dt5),
            d10_track_angle=lambda d: (d[col_track_angle].diff(10).bfill().apply(absolute_angle) / d.dt10),
        )

        flight = flight[flight[col_on_taxiway] == 1.0]
        index_0 = flight.index[0]

        if len(flight) >= self.padding_size:
            flight = flight.iloc[:self.padding_size]
        else:
            padding_length = self.padding_size - len(flight)
            padding_f = pd.DataFrame(0, index=np.arange(padding_length), columns=flight.columns)
            flight = pd.concat([flight, padding_f], ignore_index=True)

        cols_input = [col_track_angle, col_altitude, col_groundspeed, 'd10_groundspeed', 'd5_groundspeed', 'd10_track_angle',
                      'd5_track_angle']
        inputs = tf.convert_to_tensor(flight[cols_input], dtype=tf.float32)
        inputs = tf.expand_dims(inputs, axis=0)

        _, values = self.classifier(inputs).popitem()
        set_proba = values.numpy().squeeze()

        _, values = self.regressor(inputs).popitem()
        set_index = values.numpy().squeeze()

        return float(set_proba), index_0 + int(np.round(set_index, 0))
(import
    [tensorflow :as tf]
    [keras.backend [tensorflow_backend :as K]]
    [keras.callbacks [TensorBoard]]
    [keras.datasets [mnist]]
    [keras.layers [Activation Dense Flatten Reshape]]
    [keras.layers.convolutional [Conv2D]]
    [keras.models [Sequential]]
    [keras.utils [np_utils]])


(def M 10)

;; data
(defn mnist-data []
    (def [[x_train y_train] [x_test y_test]] (mnist.load_data))
    (setv x_train (/ (.astype x_train "f") 255))
    (setv x_test (/ (.astype x_test "f") 255))
    (setv y_train (np_utils.to_categorical y_train M))
    (setv y_test (np_utils.to_categorical y_test M))
    (, (, x_train y_train) (, x_test y_test)))

;; model
(defn build-model []
    (doto
        (Sequential)
        (.add (Reshape (, 28 28 1) :input_shape (, 28 28)))
        (.add (Conv2D 8 (, 5 5) :strides (, 2 2) :activation "relu"))
        (.add (Conv2D 16 (, 5 5) :strides (, 2 2) :activation "relu"))
        (.add (Flatten))
        (.add (Dense M :activation "softmax"))
        (.compile :loss "categorical_crossentropy" :optimizer "sgd" :metrics ["accuracy"])
        (.summary)))

;; learning
(with [((. (tf.Graph) as_default))]
    (def session (tf.Session ""))
    (K.set_session session)
    (K.set_learning_phase 1)

    (def [[x_train y_train] [x_test y_test]] (mnist-data))
    (def model (build-model))
    (.fit model x_train y_train
        :batch_size 30
        :epochs 10
        :validation_data (, x_test y_test)))

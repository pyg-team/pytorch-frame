import copy

from torch_frame import BaseTransform, TensorFrame


class FittableBaseTransform(BaseTransform):
    r"""An abstract base class for writing fittable transforms.

    Fittable transforms must be fitted on train data before transform.
    """
    def __call__(self, tf: TensorFrame) -> TensorFrame:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(tf))

    def fit(self, tf: TensorFrame) -> TensorFrame:
        return tf

    def forward(self, tf: TensorFrame) -> TensorFrame:
        return tf

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

from .base import Form, TransformLayer, TransformSpec


class InputLayer(TransformLayer):
    def __init__(self, form):
        self.form = form

    def forward_one(self, x, is_training):
        self.form.check(x)
        return x


class InputSpec(TransformSpec):
    def __init__(self, shape, dtype):
        self.form = Form(shape, dtype)

    def build_one(self, form):
        assert form is None
        return InputLayer(self.form), self.form

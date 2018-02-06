class Spec(object):
    def __init__(self, spatial_ndim=None):
        if spatial_ndim is None:
            x_ndim = None
        else:
            assert spatial_ndim in {0, 1, 2, 3}
            x_ndim = spatial_ndim + 2
        self._x_ndim = x_ndim

    def x_ndim(self):
        return self._x_ndim

    def batch_ndim(self):
        return self._x_ndim - 1

    def spatial_ndim(self):
        return self._x_ndim - 2

    def build_inner(self, forms):
        raise NotImplementedError

    def build(self, forms):
        assert forms
        if self._x_ndim is None:
            self._x_ndim = forms[0].batch_ndim + 1
            for form in forms[1:]:
                if form.batch_ndim + 1 != self._x_ndim:
                    self._x_ndim = None
                    break
        else:
            for form in forms:
                assert form.batch_ndim + 1 == self._x_ndim
        return self.build_inner(forms)


class TransformSpec(Spec):
    def build_transform(self, form):
        raise NotImplementedError

    def build_inner(self, forms):
        assert len(forms) == 1
        form, = forms
        layer, form = self.build_transform(form)
        return layer, [form]


class MergeSpec(Spec):
    def build_merge(self, forms):
        raise NotImplementedError

    def build_inner(self, forms):
        layer, form = self.build_merge(forms)
        return layer, [form]


class ForkSpec(Spec):
    def build_fork(self, forms):
        raise NotImplementedError

    def build_inner(self, forms):
        assert len(forms) == 1
        form, = forms
        return self.build_fork(form)


class FlexSpec(Spec):
    def build_flex(self, forms):
        raise NotImplementedError

    def build_inner(self, forms):
        return self.build_flex(forms)

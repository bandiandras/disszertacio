class Point:
    def __init__(self):
        self._x = None
        self._y = None
        self._p = None

    def __init__(self):
        self._x = None
        self._y = None
        self._p = None

    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, value):
        self._x = value
    @x.deleter
    def x(self):
        del self._x

    
    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, value):
        self._y = value
    @y.deleter
    def y(self):
        del self._y

    @property
    def p(self):
        return self._p
    @y.setter
    def p(self, value):
        self._p = value
    @y.deleter
    def p(self):
        del self._p
class Point:
    def __init__(self):
        self._x = None
        self._y = None
        self._p = None
        self._x1 = None
        self._y1 = None
        self._x2 = None
        self._y2 = None

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
    @p.setter
    def p(self, value):
        self._p = value
    @p.deleter
    def p(self):
        del self._p

    @property
    def x1(self):
        return self._x1
    @x1.setter
    def x1(self, value):
        self._x1 = value
    @x1.deleter
    def x1(self):
        del self._x1

    @property
    def x2(self):
        return self._x2
    @x2.setter
    def x2(self, value):
        self._x2 = value
    @x2.deleter
    def x2(self):
        del self._x2

    @property
    def y1(self):
        return self._y1
    @y1.setter
    def y1(self, value):
        self._y1 = value
    @y1.deleter
    def y1(self):
        del self._y1

    @property
    def y2(self):
        return self._y2
    @y2.setter
    def y2(self, value):
        self._y2 = value
    @y2.deleter
    def y2(self):
        del self._y2
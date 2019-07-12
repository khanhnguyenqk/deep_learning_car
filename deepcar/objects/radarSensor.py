from pymunk import Body, Segment, Vec2d
import pymunk

class RadarSensor:
    def __init__(self, rootPosition, angle, raceTrack, range=200):
        self.range = range
        self._angle = angle
        self._root = Vec2d(rootPosition)
        self.raceTrack = raceTrack
        self.point = None

    def _getRoot(self):
        return self._root
    def _setRoot(self, newRoot):
        self._root = newRoot
        self.updateObstaclePoint()
    root = property(_getRoot, _setRoot)

    def _getAngle(self):
        return self._angle
    def _setAngle(self, newAngle):
        self._angle = newAngle
        self.updateObstaclePoint()
    angle = property(_getAngle, _setAngle)

    def _getTail(self):
        length = Vec2d(self.range, 0)
        length.rotate(self.angle)
        return self.root + length


    def rotateDeltaAboutRoot(self, delta_radian):
        self.angle += delta_radian

    def updateObstaclePoint(self):
        self.point = None
        walls = [x.shape for x in self.raceTrack.walls]
        obstacles = []
        tail = self._getTail()
        for shape in walls:
            segmentQueryInfo = shape.segment_query(self.root, tail)
            if segmentQueryInfo.shape:
                obstacles.append(segmentQueryInfo.point)
        if not obstacles:
            return
        obstacles.sort(key=lambda x: (x - self.root).length)
        self.point = obstacles[0]
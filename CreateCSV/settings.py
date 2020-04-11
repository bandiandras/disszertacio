from enum import Enum

class TimeSeries(Enum):
    XY = 1
    X1Y1 = 2
    X2Y2 = 3

N = 512

OUTPUT_FILE_PATH_GENUINE = r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\genuine.csv"
OUTPUT_FILE_PATH_FORGERY = r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\forgery.csv"

TIME_SERIES = TimeSeries.X1Y1

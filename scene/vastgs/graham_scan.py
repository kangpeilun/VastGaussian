# Author: Peilun Kang
# Contact: kangpeilun@nefu.edu.cn
# License: Apache Licence
# Project: VastGaussian
# File: graham_scan.py
# Time: 5/16/24 2:32 PM
# Des: python实现凸包算法

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from shapely.geometry import Polygon, box
from scipy.spatial import ConvexHull


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def cross_product(a, b, c):
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def compare_angles(pivot, p1, p2):
    orientation = cross_product(pivot, p1, p2)
    if orientation == 0:
        return distance(pivot, p1) - distance(pivot, p2)
    return -1 if orientation > 0 else 1


def graham_scan(points):
    n = len(points)
    if n < 3:
        return "凸包需要至少3个点"

    pivot = min(points, key=lambda point: (point.y, point.x))
    points = sorted(points, key=lambda point: (np.arctan2(point.y - pivot.y, point.x - pivot.x), -point.y, point.x))

    stack = [points[0], points[1], points[2]]
    for i in range(3, n):
        while len(stack) > 1 and compare_angles(stack[-2], stack[-1], points[i]) > 0:
            stack.pop()
        stack.append(points[i])

    return stack


def plot_convex_hull(points, convex_hull, x, y):
    plt.figure()
    plt.scatter([p.x for p in points], [p.y for p in points], color='b', label="所有点")

    # 绘制凸包
    plt.plot([p.x for p in convex_hull] + [convex_hull[0].x], [p.y for p in convex_hull] + [convex_hull[0].y],
             linestyle='-', color='g', label="篱笆边")

    for i in range(len(convex_hull)):
        plt.plot([convex_hull[i].x, convex_hull[(i + 1) % len(convex_hull)].x],
                 [convex_hull[i].y, convex_hull[(i + 1) % len(convex_hull)].y], linestyle='-', color='g')

    plt.plot(x, y)

    plt.show()


def run_graham_scan(points, W, H):
    """获取8个点围成的区域的凸包
    :param points 8个角点投影后的坐标
    :param W 图像宽度
    :param H 图像高度
    :return 凸包的点集 [x, y]
    """
    # points = [Point(point[0], point[1]) for point in points]
    # convex_hull = graham_scan(points)
    points = np.array(points)
    convex_hull = ConvexHull(np.array(points))
    # convex_hull_polygon = Polygon([(point[0], point[1]) for point in convex_hull])

    convex_hull_list = []
    # plt.plot(points[:, 0], points[:, 1], 'o')
    for i, j in zip(convex_hull.simplices, convex_hull.vertices):
        # plt.plot(points[i, 0], points[i, 1], 'k-')
        convex_hull_list.append(points[j])

    convex_hull_polygon = Polygon(convex_hull_list)
    image_bounds = box(0, 0, W, H)
    # x = [0, W, W, 0, 0]
    # y = [0, 0, H, H, 0]
    # plt.plot(x, y)
    # plt.show()
    # plot_convex_hull(points, convex_hull, x, y)
    # 计算凸包与图像边界的交集
    intersection = convex_hull_polygon.intersection(image_bounds)
    image_area = W * H  # 图像面积
    # 计算交集面积占图像面积的比例
    intersection_rate = intersection.area / image_area

    # print("intersection_area: ", intersection.area, " image_area: ", image_area, " intersection_rate: ", intersection_rate)
    return {
        "intersection_area": intersection.area,
        "image_area": image_area,
        "intersection_rate": intersection_rate,
    }

# -*- coding: utf-8 -*-
import arcpy
import numpy as np
import matplotlib.pyplot as plt
import os

class Toolbox(object):
    def __init__(self):
        self.label = "Hotspot Analysis Toolbox"
        self.alias = "hotspot_analysis_toolbox"
        self.tools = [HotspotDetectionTool]

class HotspotDetectionTool(object):
    def __init__(self):
        self.label = "Bernoulli Hotspot Detection Tool"
        self.description = "Detects hotspots in randomly generated data points using statistical likelihood methods."
        self.canRunInBackground = False

    def getParameterInfo(self):
        # Define parameters for ArcGIS Tool
        in_features = arcpy.Parameter(
            displayName="Input Point Layer",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")

        label_field = arcpy.Parameter(
            displayName="Label Field",
            name="label_field",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        label_field.parameterDependencies = ["in_features"]

        out_patterns = arcpy.Parameter(
            displayName="Output Pattern Feature Class",
            name="out_patterns",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output")

        run_monte_carlo = arcpy.Parameter(
            displayName="Run Monte Carlo Simulation",
            name="run_monte_carlo",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input")
        run_monte_carlo.value = False  # Default is set to False

        p_value = arcpy.Parameter(
            displayName="P-Value Significance Threshold",
            name="p_value",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")
        p_value.value = 0.05  # Default p-value significance threshold

        return [in_features, label_field, out_patterns, run_monte_carlo, p_value]

    def execute(self, parameters, messages):
        # Retrieve input values
        in_features = parameters[0].valueAsText
        label_field = parameters[1].valueAsText
        out_patterns = parameters[2].valueAsText
        run_monte_carlo = parameters[3].value  # Boolean to decide if Monte Carlo should run
        p_value_threshold = parameters[4].value  # P-value significance threshold

        # Load points and labels into NumPy arrays
        with arcpy.da.SearchCursor(in_features, ["SHAPE@XY", label_field]) as cursor:
            points = []
            labels = []
            for row in cursor:
                points.append(row[0])
                labels.append(row[1])

        X_loc_r = np.array(points)
        y_r = np.array(labels).reshape(-1, 1)

        # Combine coordinates and labels
        sample_data = np.concatenate([X_loc_r, y_r], axis=1)

        # Detect hotspots
        patterns = []
        data = np.copy(sample_data)
        def takeLast(elem):
            return elem[2]
        for i in range(3):
            result_2P = self.findHotSpot2P(data, 0, "log", 1)
            result_2P.sort(key=takeLast, reverse=True)
            if run_monte_carlo:
                MC_P, MC_count = self.monteCarloP(100, data, result_2P[0], 1)
                if MC_P <= p_value_threshold:
                    patterns.append(result_2P[0])
                    data = self.delete_pattern(data, result_2P[0])
                else:
                    break
            else:
                patterns.append(result_2P[0])
                data = self.delete_pattern(data, result_2P[0])

        # Create output patterns feature class

        # Split `out_patterns` into path and name
        out_path, out_name = os.path.split(out_patterns)

        # Create the feature class with correct parameters
        arcpy.CreateFeatureclass_management(out_path=out_path, out_name=out_name, geometry_type="POLYGON")
        arcpy.AddField_management(out_patterns, "Radius", "DOUBLE")
        arcpy.AddField_management(out_patterns, "Score", "DOUBLE")

        with arcpy.da.InsertCursor(out_patterns, ["SHAPE@", "Radius", "Score"]) as cursor:
            for pattern in patterns:
                (x, y), radius, score = pattern[0], pattern[1], pattern[2]
                circle = arcpy.PointGeometry(arcpy.Point(x, y)).buffer(radius)
                cursor.insertRow([circle, radius, score])

        messages.addMessage("Hotspot Detection completed successfully.")



    @staticmethod
    def distPnts(pnt1, pnt2):
        return np.sqrt((pnt2[0] - pnt1[0]) ** 2 + (pnt2[1] - pnt1[1]) ** 2)

    @staticmethod
    def countPnts_circle(dataset, center, radius, sampletotal=10):
        distance = np.sqrt(np.sum((dataset[:, :2] - center) ** 2, axis=1))
        numTotal = sampletotal * np.sum(distance <= radius)
        numHot = np.sum((distance <= radius).astype(int) * np.rint(sampletotal * dataset[:, 2]))
        return numTotal, numHot

    @staticmethod
    def countPnts(dataset, sampletotal=10):
        numTotal = sampletotal * dataset.shape[0]
        numHot = np.sum(np.rint(sampletotal * dataset[:, 2]))
        return numTotal, numHot

    def likelihood_PlQ(self, numTotalinCircle, numHotinCircle, numTotal, numHot, mode="log"):
        numTotaloutCircle = numTotal - numTotalinCircle
        numHotoutCircle = numHot - numHotinCircle
        if mode == "log":
            L_hotinLog = np.log(numHotinCircle / numTotalinCircle + 1e-8) * numHotinCircle
            L_coldinLog = np.log(1 - numHotinCircle / numTotalinCircle + 1e-8) * (numTotalinCircle - numHotinCircle)
            L_hotoutLog = np.log(numHotoutCircle / numTotaloutCircle + 1e-8) * numHotoutCircle
            L_coldoutLog = np.log(1 - numHotoutCircle / numTotaloutCircle + 1e-8) * (numTotaloutCircle - numHotoutCircle)
            return L_hotinLog + L_coldinLog + L_hotoutLog + L_coldoutLog

    def likelihood_PsQ(self, numTotal, numHot, mode="log"):
        if (mode == "log"):
            L_hotLog = np.log(numHot/numTotal+1e-8) * numHot
            L_coldLog = np.log(1 - numHot/numTotal+1e-8) * (numTotal - numHot)

            L_log = L_hotLog + L_coldLog
            return L_log

    def findHotSpot2P(self, sampleData, LR_threshold, mode="log", sampletotal=10):
        numTotal, numHot = self.countPnts(sampleData, sampletotal)
        hotSpotList = []
        for idx in range(sampleData.shape[0]):
            for radius in range(500, 1000, 100):
                center = sampleData[idx, :2]
                if not self.is_inside_boundary(center, radius):
                    continue
                numTotalinCircle, numHotinCircle = self.countPnts_circle(sampleData, center, radius, sampletotal)
                if numTotalinCircle == 0 or numTotal - numTotalinCircle == 0:
                    continue
                if self.likelihood_I(numTotalinCircle, numHotinCircle, numTotal, numHot):
                    L_PlQ = self.likelihood_PlQ(numTotalinCircle, numHotinCircle, numTotal, numHot, mode)
                    L_PsQ = self.likelihood_PsQ(numTotal, numHot, mode)
                    LR = L_PlQ - L_PsQ
                    if LR >= LR_threshold:
                        hotSpotList.append([center, radius, LR, numHotinCircle])
        return hotSpotList

    @staticmethod
    def likelihood_I(numTotalinCircle, numHotinCircle, numTotal, numHot):
        p = numHotinCircle / numTotalinCircle
        q = (numHot - numHotinCircle) / (numTotal - numTotalinCircle)
        return p > q

    def monteCarloP(self, numItrtn, data, result, sampletotal=10):
        count = 0
        for _ in range(numItrtn):
            MCData = self.generateRandomData_2(data)
            MCResult = self.findHotSpot2P(MCData, 0, "log", sampletotal)
            if not MCResult:
                continue
            if MCResult[0][2] > result[2]:
                count += 1
        return count / numItrtn, count

    @staticmethod
    def is_inside_boundary(point, pattern_size, x_min=0, x_max=4096):
        y_min = x_min
        y_max = x_max
        return not (
            point[0] <= x_min + pattern_size / 1.2 or
            point[1] <= y_min + pattern_size / 1.2 or
            point[0] >= x_max - pattern_size / 1.2 or
            point[1] >= y_max - pattern_size / 1.2
        )

    @staticmethod
    def delete_pattern(dataset, pattern):
        center, radius = pattern[0], pattern[1]
        distance = np.sqrt(np.sum(np.square(dataset[:, :2] - center), axis=1))
        idx = np.where(distance <= radius)
        return np.delete(dataset, idx, 0)

    def generateRandomData_2(self, sampleData):
        loc = sampleData[:, :2]
        c = sampleData[:, 2:]
        loc_p = np.random.permutation(loc)
        return np.concatenate([loc_p, c], axis=-1)

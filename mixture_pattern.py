import arcpy
import numpy as np
import os

class Toolbox(object):
    def __init__(self):
        self.label = "Hotspot Detection Toolbox"
        self.alias = "hotspot_detection_toolbox"
        self.tools = [HotspotDetectionTool]

class HotspotDetectionTool(object):
    def __init__(self):
        self.label = "Mixture Pattern Detection Tool"
        self.description = "Detects mixture patterns based on class labels in a point layer."
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
        X = np.array(points)
        y_r = np.array(labels).astype(int)

        # Set constants
        NUM_CLASS = 21
        y_r_o = np.eye(NUM_CLASS)[y_r]  # One-hot encoding using NumPy
        X_MIN, X_MAX = 0, 4096  # Boundaries
        Start_SIZE = 100

        def get_top_SMI(X, y, score_type='se_exact', rate=1):
            pattern_list = []
            for size in range(500, 1000, 100):
                for i in range(X.shape[0]):
                    if i % rate == 0:
                        pset = get_pattern_points(X, i, size)
                        num = y[pset].shape[0]
                        if num != 1:
                            pattern_list.append([X[i, 0], X[i, 1], size, get_SMI_score(y[pset], score_type)])
            pattern_list = np.array(pattern_list)
            sorted_ids = np.argsort(pattern_list[:, -1])[::-1]
            return pattern_list, sorted_ids

        def is_inside_boundary(point, pattern_size, x_min=0, x_max=4096):
            y_min = x_min
            y_max = x_max
            return not (point[0] <= x_min + pattern_size / 1.2 or 
                        point[1] <= y_min + pattern_size / 1.2 or 
                        point[0] >= x_max - pattern_size / 1.2 or 
                        point[1] >= y_max - pattern_size / 1.2)

        def get_pattern_points(X, i, size):
            point = X[i]
            distances = np.sqrt(np.sum((X - point) ** 2, axis=1))
            pset = np.where(distances <= size)[0]
            return pset

        def get_SMI_score(pt, score_type):
            if score_type == 'se_prob':
                return se_prob(pt)
            return 0

        def se_prob(pt):
            pt = np.sum(pt, axis=0)
            pt = pt / (np.sum(pt) + 1e-8)
            log_pt = np.log(pt + 1e-8)
            return -np.sum(pt * log_pt)

        def delete_pattern(loc, pattern):
            center = pattern[:2]
            radius = pattern[2]
            distance = np.sqrt(np.sum((loc - center) ** 2, axis=1))
            return np.where(distance > radius)

        def monteCarloP(numItrtn, X_loc_r_cp, y_r_o_cp, result):
            count = 0
            np.random.seed(10)
            for _ in range(numItrtn):
                X_loc_r_rd, y_r_o_rd = generateRandomData_2(X_loc_r_cp, y_r_o_cp)
                pattern_list_exact, sorted_ids_exact = get_top_SMI(X_loc_r_rd, y_r_o_rd, score_type='se_prob')
                top_pattern = pattern_list_exact[sorted_ids_exact[0]]
                if top_pattern[3] > result[3]:
                    count += 1
            return count / numItrtn, count

        def generateRandomData_2(loc, c):
            loc_p = np.random.permutation(loc)
            return loc_p, c

        patterns = []
        X_loc_r_cp = X.copy()
        y_r_o_cp = y_r_o.copy()


        for _ in range(3):
            pattern_list_exact, sorted_ids_exact = get_top_SMI(X_loc_r_cp, y_r_o_cp, score_type='se_prob', rate=1)
            pattern_sorted = pattern_list_exact[sorted_ids_exact]
            top_pattern = pattern_sorted[0]

            if run_monte_carlo:
                MC_P, MC_count = monteCarloP(100, X_loc_r_cp, y_r_o_cp, top_pattern)
                if MC_P <= p_value_threshold:
                    patterns.append(top_pattern)
                    select_idx = delete_pattern(X_loc_r_cp, top_pattern)
                    X_loc_r_cp = X_loc_r_cp[select_idx]
                    y_r_o_cp = y_r_o_cp[select_idx]
                else:
                    break
            else:
                patterns.append(top_pattern)
                select_idx = delete_pattern(X_loc_r_cp, top_pattern)
                X_loc_r_cp = X_loc_r_cp[select_idx]
                y_r_o_cp = y_r_o_cp[select_idx]


        # Create output patterns feature class

        # Split `out_patterns` into path and name
        out_path, out_name = os.path.split(out_patterns)

        # Create the feature class with correct parameters
        arcpy.CreateFeatureclass_management(out_path=out_path, out_name=out_name, geometry_type="POLYGON")
        arcpy.AddField_management(out_patterns, "Radius", "DOUBLE")
        arcpy.AddField_management(out_patterns, "Score", "DOUBLE")

        with arcpy.da.InsertCursor(out_patterns, ["SHAPE@", "Radius", "Score"]) as cursor:
            for pattern in patterns:
                x, y, radius, score = pattern
                circle = arcpy.PointGeometry(arcpy.Point(x, y)).buffer(radius)
                cursor.insertRow([circle, radius, score])

        messages.addMessage("Hotspot Detection completed successfully.")

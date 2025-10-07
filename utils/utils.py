import numpy as np

# def save_parameters_to_json(json_fname):
#     params = {}
#     params["Window size"] = list(gvxr.getWindowSize()),
#     params["Source"] = {
#         "Position": list(gvxr.getSourcePosition("mm")) + ["mm"],
#         "Shape" : "PARALLEL",
#         "Beam":list({
#             "Energy": energy,
#             "Unit": energy_units
#         })
#     }
#     params["Detector"] = {
#         "Position" : list(gvxr.getDetectorPosition("mm")) + ["mm"],
#         "UpVector" : list(gvxr.getDetectorUpVector()),
#         "RightVector" : list(gvxr.getDetectorRightVector()),
#         "NumberOfPixels" : list(gvxr.getDetectorNumberOfPixels()),
#         "Size" : list(gvxr.getDetectorSize("mm")) + ["mm"]

#     }
#     params["Scan"] = {
#         "OutFolder": sub_folder,
#         "NumberOfProjections": len(xray_image_set),
#         "AngleStep": step,
#         "StartAngle": start,
#         "FinalAngle": stop,
#         "IncludeLastAngle": True,  
#         "Flat-Field Correction": False,
#         "CentreOfRotation": list(gvxr.getCentreOfRotationPositionCT("mm")) + ["mm"],
#         "RotationAxis": list(rotation_axis)
#     }
#     print(params)
#     with open(json_fname, "w") as file:
#         json.dump(params, file, indent=4)


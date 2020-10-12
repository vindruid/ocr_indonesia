import os
cwd = os.getcwd() #it is say where this config file located

# create args
cfg = dict()

# args for boundingBoxesDetectron2
cfg['boundingBoxesDetectron2'] = {
        "model_bb_sim" : "model_weights/box_detector_SIM__faster_rcnn_R_101_FPN_3x.pth", 
        "model_bb_ktp" : "model_weights/box_detector_KTP__faster_rcnn_R_101_FPN_3x.pth",
        "num_class_sim" : 3,
        "num_class_ktp" : 2,
        "thresh" : 0.7,
        "cuda" : 1,
        "model_zoo_cfg" : "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    }


# args for numericalDetectron2
cfg['numericalDetectron2'] = {
        "list_model_zoo" : ["COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"],
#         "list_model_path" : ["model_weights/char_detector_NIK_NOSIM_acc98__X101_custom_mapper_aug.pth"],
#         "name_class" : "0123456789",
#         "num_class" : 10,
        "list_model_path" : ["model_weights/char_detector_NIK_NOSIM_EXP_acc946__X101_custom_mapper_aug.pth"],
        "name_class" : "0123456789-",
        "num_class" : 11,
        "threshold" : 0.65,
        "list_cuda" : [1],
        "model_weights" : [1],  

    }

# args for alphabeticalDetectron2
cfg['alphabeticalDetectron2'] = {
        "list_model_zoo" : [
            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
            "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"], 
        "list_model_path" : [
            'model_weights/model_final_ktp_sim-fix_rgb_x101.pth',
            'model_weights/model_final_ktp_sim-grayscale.pth',
            'model_weights/model_final_ktp_sim-fix_rgb_rotated_X101_custom_mapper.pth',
            'model_weights/model_final_ktp_sim-fix_rgb_rotated_X101_custom_mapper_many_augmentation.pth'], # place the best predictor in the last list, so it can be called using predict_by_last_predictor(img) function
        "threshold" : 0.65,
        "list_cuda" : [2, 3],
        "num_class" : 27,
        "model_weights" : [0.8, 0.83, 0.86, 0.87],
        "name_class" : " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    }
    


cfg['drawOCR'] = {
    "COLORS" : [(0, 120, 0), (120, 0, 50), (0, 100, 100)]
}

# Standarize model location with current dir
cfg['numericalDetectron2']["list_model_path"] = [os.path.join(cwd, item_path) for item_path in cfg['numericalDetectron2']["list_model_path"]]
cfg['alphabeticalDetectron2']["list_model_path"] = [os.path.join(cwd, item_path) for item_path in cfg['alphabeticalDetectron2']["list_model_path"]]
cfg['boundingBoxesDetectron2']["model_bb_sim"] = os.path.join(cwd, cfg['boundingBoxesDetectron2']["model_bb_sim"])
cfg['boundingBoxesDetectron2']["model_bb_ktp"] = os.path.join(cwd, cfg['boundingBoxesDetectron2']["model_bb_ktp"]) 



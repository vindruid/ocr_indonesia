from gdrive_utils import download_file
import os
import re
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image

from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


def check_area(b):
    x01, y01, x11, y11 = b

    return (x11 - x01) * (y11 - y01)


def check_roi(b1, b2, div='usemin'):
    x01, y01, x11, y11 = b1
    x02, y02, x12, y12 = b2
    xb = max(x01, x02)
    xt = min(x11, x12)
    yb = max(y01, y02)
    yt = min(y11, x12)

    if isinstance(div, str):
        division = min(((x11 - x01) * (y11 - y01)),
                       ((x12 - x02) * (y12 - y02)))
    else:
        if div == 1:
            division = (x11 - x01) * (y11 - y01)
        elif div == 2:
            division = (x12 - x02) * (y12 - y02)
        else:
            division = min(((x11 - x01) * (y11 - y01)),
                           ((x12 - x02) * (y12 - y02)))

    if division <= 0:
        return 0

    else:
        if ((xt - xb) < 0) | ((yt - yb) < 0):
            roi = 0
        else:
            Area = (xt - xb) * (yt - yb)

            roi = max(0, Area)
            roi = min(1, roi / division)

        return roi


def nms_boxes(boxes_class_score, thresh=0.45):
    removed = []

    for j in range(len(boxes_class_score)):
        for k in range(len(boxes_class_score)):
            if j > k:
                if check_roi(boxes_class_score[j][0:4], boxes_class_score[k][0:4]) > thresh:
                    scorek = boxes_class_score[k][5]
                    scorej = boxes_class_score[j][5]

                    if scorek > scorej:
                        removed.append(j)

                    elif scorek == scorej:
                        if check_area(boxes_class_score[j][0:4]) > check_area(boxes_class_score[k][0:4]):
                            removed.append(k)
                        else:
                            removed.append(j)
                    else:
                        removed.append(k)

    new_box = np.delete(boxes_class_score, list(set(removed)), axis=0)

    return new_box


def load_predictor(model_zoo_cfg="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml", num_class=27,
                   model_path=os.path.join('output_model', 'model_final.pth'), thresh=0.65, cuda=2, use_cuda=True):
    if not os.path.exists(model_path):
        download_file(model_path)
    if use_cuda:
        with torch.cuda.device(cuda):

            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(model_zoo_cfg))
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class
            cfg.MODEL.WEIGHTS = model_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
            cfg.CUDA = "cuda:" + str(cuda)
            cfg.INPUT.FORMAT = "RGB"
            if "custom_mapper" in model_path:
                predictor = Predictor_Modified(cfg)
            else:
                predictor = DefaultPredictor(cfg)
    else:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_zoo_cfg))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        cfg.MODEL.DEVICE = 'cpu'
        cfg.INPUT.FORMAT = "RGB"
        if "custom_mapper" in model_path:
            predictor = Predictor_Modified(cfg)
        else:
            predictor = DefaultPredictor(cfg)

    return predictor


def list_load_predictor(list_model_zoo_cfg, list_model_path, thresh=0.65, list_cuda=[0, 1, 2, 3], num_class=27, use_cuda=True):
    list_predictor = []
    for i, (model_zoo_cfg, model_path) in enumerate(list(zip(list_model_zoo_cfg, list_model_path))):
        predictor = load_predictor(model_zoo_cfg=model_zoo_cfg, num_class=num_class,
                                   model_path=model_path, thresh=thresh, cuda=list_cuda[i % len(list_cuda)], use_cuda=use_cuda)
        list_predictor.append(predictor)

    return list_predictor


def easypredict(image, bBoxDet, numDet, alphaDet, input_type='ktp', ensemble_method="weighted_hardvote_word"):
    # detect boundingboxes
    crops, _, _ = bBoxDet.predict(image, input_type=input_type)
    
    result = {}
    result['input_type'] = input_type
    
    # detect ID number
    if crops[0] is not None: 
        result['id'] = numDet.predict_ensemble(crops[0]).get(ensemble_method)
        
    # detect Name
    if crops[1] is not None: 
        result['name'] = alphaDet.predict_ensemble(crops[1]).get(ensemble_method)

    # detect Expired date
    if crops[2] is not None: 
        result['expdate'] = numDet.predict_ensemble(crops[2]).get(ensemble_method)
    
    return result


class Predictor_Modified(object):

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class numericalDetectron2(object):

    def __init__(self, args={}):
        self.list_model_zoo_cfg = args.get(
            "list_model_zoo", ["COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"])
        self.list_model_path = args.get(
            "list_model_path", ["char_detector_NIK_NOSIM_EXP_acc933__X101_custom_mapper_aug.pth"])
        self.threshold = args.get("threshold", 0.65)
        self.list_cuda = args.get("list_cuda", [1])
        self.num_class = args.get("num_class", 11)
        self.name_class = args.get("name_class", "0123456789-")
        self.model_weights = args.get("model_weights", [1])
        self.list_predictor = list_load_predictor(
            self.list_model_zoo_cfg, self.list_model_path, thresh=self.threshold, list_cuda=self.list_cuda, num_class=self.num_class)

        self.dataframe = pd.read_csv(os.path.join(
            "assets", "data_kode_wilayah.csv"), index_col=0)

    def predict_by_last_predictor(self, img):
        # predict using last predictor in list_predictor
        return self.predict(self.list_predictor[-1], img)[0]

    def predict(self, predictor, img):
        im_resized = cv2.resize(
            img, (int(img.shape[1] * 100 / img.shape[0]), 100))
        im_resized = Image.fromarray(im_resized[:, :, ::-1])
        im_resized = im_resized.convert('L')
        im_resized = im_resized.convert('RGB')
        im_resized = np.asarray(im_resized)[:, :, ::-1]
        outputs = predictor(im_resized)
        predictions = outputs["instances"].to("cpu")
        pred_class = predictions.pred_classes.numpy()
        boxes = predictions.pred_boxes.tensor.numpy().astype(int)
        scores = (predictions.scores * 100).type(torch.int32)
        boxes_n_class = np.concatenate(
            (boxes, pred_class.reshape(-1, 1)), axis=1)
        boxes_class_score = np.concatenate(
            (boxes_n_class, scores.reshape(-1, 1)), axis=1)
        boxes_class_score = np.sort(boxes_class_score.view(
            "i8,i8,i8,i8,i8,i8"), order=['f0'], axis=0).view(np.int)
        boxes_class_score = nms_boxes(boxes_class_score)
        final_pred = [self.name_class[char_id] for char_id in boxes_class_score[:, 4]]
        final_pred = "".join(final_pred)

        return final_pred, boxes_class_score

    def char_voting(self, list_array, mode='hard', weighted=False):
        lengths = []
        for array in list_array:
            length = array.shape[0]
            lengths.append(length)

        char_voted = []
        if mode == 'hard':
            if len(set(lengths)) == 1:
                for i in range(lengths[0]):
                    char = [x[i][4] for x in list_array]
                    if weighted:
                        char_weighted_idx = self.weighted_voting(char)
                        char_voted.append(str(char_weighted_idx))
                    else:
                        values, counts = np.unique(char, return_counts=True)
                        maxidx = np.argmax(counts)
                        charidx = values[maxidx]
                        char_voted.append(str(charidx))

            else:
                list_result = []
                for bcs in list_array:
                    final_pred = bcs[:, 4]
                    final_pred = "".join(final_pred)
                    list_result.append(final_pred)

                if weighted:
                    fulltext_voting = self.weighted_voting(list_result)

                else:
                    values, counts = np.unique(list_result, return_counts=True)
                    maxidx = np.argmax(counts)
                    fulltext_voting = values[maxidx]

                return fulltext_voting

        if mode == 'soft':
            if not weighted:
                model_weights = [1] * len(list_array)

            else:
                model_weights = self.model_weights

            if len(set(lengths)) == 1:
                for i in range(lengths[0]):
                    char = {}
                    char_value = [(x[i][4], x[i][5] * model_weights[ix])
                                  for ix, x in enumerate(list_array)]
                    for c, v in char_value:
                        if c in char:
                            char[c] += v
                        else:
                            char[c] = v
                    charidx = max(char, key=char.get)
                    char_voted.append(str(charidx))

            else:
                mean_score = []
                for i, array in enumerate(list_array):
                    score = list_array[0][:, 5].mean() * model_weights[i]
                    mean_score.append(score)
                maxidx = np.argmax(mean_score)
                bcs = list_array[maxidx]
                char_voted = bcs[:, 4].astype(str)

        return "".join(char_voted)

    def weighted_voting(self, list_result):
        result = {}
        for i in range(len(list_result)):
            w = self.model_weights[i]
            if list_result[i] not in result:
                result[list_result[i]] = [w]
            else:
                result[list_result[i]].append(w)
        for key in result:
            result[key] = np.sum(result[key])

        return max(result, key=result.get)

    def softvote_words(self, list_array):
        scores = []
        for arr in list_array:
            mean_score = arr[:, 5].mean()
            scores.append(mean_score)

        return np.argmax(scores)

    def predict_ensemble(self, im):
        list_result = []
        list_array = []

        for predictor in self.list_predictor:
            result, array = self.predict(predictor, im)
            list_result.append(result)
            list_array.append(array)

        # voting from total match
        values, counts = np.unique(list_result, return_counts=True)
        maxidx = np.argmax(counts)
        word_hardvote = values[maxidx]

        # voting from total mean score
        idx = self.softvote_words(list_array)
        word_softvote = list_result[idx]

        # voting with weight
        word_weight_hardvote = self.weighted_voting(list_result)

        # voting char by char
        char_hardvote = self.char_voting(list_array)
        char_softvote = self.char_voting(list_array, mode='soft')
        char_weight_hardvote = self.char_voting(list_array, weighted=True)
        char_weight_softvote = self.char_voting(
            list_array, mode='soft', weighted=True)

        dict_ensemble = dict()
        names = ['hardvote_word', 'hardvote_char', 'softvote_char', 'softvote_word',
                 'weighted_hardvote_word', 'weighted_hardvote_char', 'weighted_softvote_char']
        results = [word_hardvote, char_hardvote, char_softvote, word_softvote,
                   word_weight_hardvote, char_weight_hardvote, char_weight_softvote]
        for nam, res in list(zip(names, results)):
            dict_ensemble[nam] = res

        return dict_ensemble

    def parse_nik(self, input_nik):
        # mengubah nomorktp menjadi string
        nomorktp = str(input_nik)

        # mengecek apakah panjang nomor KTP sesuai
        if len(nomorktp) != 16:
            return {"Message": "Please retake your KTP !"}

        else:
            try:
                # mengecek apakah kode lokasi ada pada data yang sudah kita scrap dari wikipedia :)
                hasil = self.dataframe.loc[self.dataframe['KodeWilayah'] == int(
                    nomorktp[:6]), ['Provinsi', 'DaerahTingkatDua', "Kecamatan"]].values[0]
                prov = hasil[0]
                wil = hasil[1]
                kec = hasil[2]

            except IndexError:
                # jika kode lokasi tidak ditemukan maka akan menampilkan `input salah`
                prov = 'Input Salah'
                wil = 'Input Salah'
                kec = 'Input Salah'

            tanggal = int(nomorktp[6:8])
            bulan = nomorktp[8:10]
            tahun = int(nomorktp[10:12])

            # mengubah dua digit tahun menjadi empat digit (NOTED : rumus ini akan berubah seiring berjalannya waktu)
            if tahun <= 9:
                tahun = '200' + str(tahun)
            else:
                tahun = '19' + str(tahun)

            # mencari gender dan tanggal lahir
            if (tanggal > 31) & (tanggal < 72):
                gender = 'Perempuan'
                tanggal = tanggal - 40
                tanggal_lahir = str(tanggal) + '-' + bulan + '-' + tahun
            elif tanggal > 71:
                gender = 'Input Salah'
                tanggal_lahir = 'Input Salah'
            else:
                gender = 'Laki-Laki'
                tanggal_lahir = str(tanggal) + '-' + bulan + '-' + tahun

            return {"Provinsi": prov, "Kabupaten/Kota": wil, "Kecamatan": kec, "Jenis Kelamin": gender, "Tanggal Lahir": tanggal_lahir}


class boundingBoxesDetectron2(object):
    def __init__(self, args={}):
        self.model_weights_sim = args.get("model_bb_sim", os.path.join(
            "model_weights", "box_detector_SIM__faster_rcnn_R_101_FPN_3x.pth"))
        self.model_weights_ktp = args.get("model_bb_ktp", os.path.join(
            "model_weights", "box_detector_KTP__faster_rcnn_R_101_FPN_3x.pth"))
        self.num_class_sim = args.get("num_class_sim", 2)
        self.num_class_ktp = args.get("num_class_ktp", 2)
        self.thresh = args.get("thresh", 0.7)
        self.model_zoo_cfg = args.get(
            "model_zoo_cfg", "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        self.cuda = args.get("cuda", 1)
        self.load_model()

    def load_model(self):

        # cfg for detectron2 detect nama and No.SIM
        self.bb_predictor_sim = load_predictor(model_zoo_cfg=self.model_zoo_cfg, num_class=self.num_class_sim,
                                               model_path=self.model_weights_sim, thresh=self.thresh, cuda=self.cuda)

        # cfg for detectron2 detect nama and NIK
        self.bb_predictor_ktp = load_predictor(model_zoo_cfg=self.model_zoo_cfg, num_class=self.num_class_ktp,
                                               model_path=self.model_weights_ktp, thresh=self.thresh, cuda=self.cuda)

    def predict(self, im, input_type='ktp'):
        if input_type == "ktp":
            outputs = self.bb_predictor_ktp(im)
        else:
            outputs = self.bb_predictor_sim(im)
        predictions = outputs["instances"].to("cpu")
        H, W = predictions.image_size
        pred_class = predictions.pred_classes.numpy()[:3]
        boxes = predictions.pred_boxes.tensor.numpy().astype(int)[:3]
        scores = predictions.scores.numpy()

        crops = dict({0: None, 1: None, 2: None})
        bboxes = dict({0: None, 1: None, 2: None})
        labels = dict({0: None, 1: None, 2: None})

        for cls_, box in list(zip(pred_class, boxes)):
            x1, y1, x2, y2 = box
            crop = im[y1:y2, x1:x2]

            if cls_ == 0:
                if input_type == 'ktp':
                    label = 'nik'

                else:
                    label = 'no.sim'

                crops[0] = crop
                bboxes[0] = box
                labels[0] = label

            elif cls_ == 1:
                label = 'nama'
                crops[1] = crop
                bboxes[1] = box
                labels[1] = label

            elif cls_ == 2:
                label = 'exp.date'
                crops[2] = crop
                bboxes[2] = box
                labels[2] = label

        return crops, bboxes, labels


class alphabeticalDetectron2(object):

    def __init__(self, args={}):
        self.list_model_zoo_cfg = args.get(
            "list_model_zoo", ["COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"])
        self.list_model_path = args.get(
            "list_model_path", ["model_final_ktp_sim-fix_rgb_rotated_X101_custom_mapper.pth"])
        self.threshold = args.get("threshold", 0.65)
        self.list_cuda = args.get("list_cuda", [2, 3])
        self.num_class = args.get("num_class", 27)
        self.model_weights = args.get("model_weights", [1])
        self.name_class = args.get("name_class", " ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.list_predictor = list_load_predictor(
            self.list_model_zoo_cfg, self.list_model_path, thresh=self.threshold, list_cuda=self.list_cuda, num_class=self.num_class)

    def predict_by_last_predictor(self, img):
        # predict using last predictor in list_predictor
        return self.predict(self.list_predictor[-1], img)[0]

    def rule_O(self, text):
        consonant = 'bcdfghjklmnpqrstvwxyz'.upper()
        consonant = [c for c in consonant]
        new_text = []
        for i, t in enumerate(text):
            if t == "Q":
                if (i != 0) & (i != len(text)-1):
                    if (text[i-1] in consonant) & (text[i+1] in consonant):
                        t = "O"
            new_text.append(t)

        return "".join(new_text)

    def predict(self, predictor, img):

        im_resized = cv2.resize(
            img, (int(img.shape[1] * 100 / img.shape[0]), 100))
        im_resized = Image.fromarray(im_resized[:, :, ::-1])
        im_resized = im_resized.convert('L')
        im_resized = im_resized.convert('RGB')
        im_resized = np.asarray(im_resized)[:, :, ::-1]
        outputs = predictor(im_resized)
        predictions = outputs["instances"].to("cpu")
        pred_class = predictions.pred_classes.numpy()
        boxes = predictions.pred_boxes.tensor.numpy().astype(int)
        scores = (predictions.scores * 100).type(torch.int32)
        boxes_n_class = np.concatenate(
            (boxes, pred_class.reshape(-1, 1)), axis=1)
        boxes_class_score = np.concatenate(
            (boxes_n_class, scores.reshape(-1, 1)), axis=1)
        boxes_class_score = np.sort(boxes_class_score.view(
            "i8,i8,i8,i8,i8,i8"), order=['f0'], axis=0).view(np.int)
        boxes_class_score = nms_boxes(boxes_class_score)
        final_pred = [self.name_class[char_id]
                      for char_id in boxes_class_score[:, 4]]
        final_pred = "".join(final_pred)
        final_pred = self.rule_O(self.cleaning_string(final_pred))

        return final_pred, boxes_class_score

    def char_voting(self, list_array, mode='hard', weighted=False):
        lengths = []
        for array in list_array:
            length = array.shape[0]
            lengths.append(length)

        char_voted = []
        if mode == 'hard':
            if len(set(lengths)) == 1:
                for i in range(lengths[0]):
                    char = [x[i][4] for x in list_array]
                    if weighted:
                        char_weighted_idx = self.weighted_voting(char)
                        char_voted.append(self.name_class[char_weighted_idx])
                    else:
                        values, counts = np.unique(char, return_counts=True)
                        maxidx = np.argmax(counts)
                        charidx = values[maxidx]
                        char_voted.append(self.name_class[charidx])

            else:
                list_result = []
                for bcs in list_array:
                    final_pred = [self.name_class[char_id]
                                  for char_id in bcs[:, 4]]
                    final_pred = "".join(final_pred)
                    final_pred = self.rule_O(final_pred)
                    list_result.append(self.cleaning_string(final_pred))

                if weighted:
                    fulltext_voting = self.weighted_voting(list_result)

                else:
                    values, counts = np.unique(list_result, return_counts=True)
                    maxidx = np.argmax(counts)
                    fulltext_voting = values[maxidx]

                return fulltext_voting

        if mode == 'soft':
            if not weighted:
                model_weights = [1] * len(list_array)

            else:
                model_weights = self.model_weights

            if len(set(lengths)) == 1:
                for i in range(lengths[0]):
                    char = {}
                    char_value = [(x[i][4], x[i][5] * model_weights[ix])
                                  for ix, x in enumerate(list_array)]
                    for c, v in char_value:
                        if c in char:
                            char[c] += v
                        else:
                            char[c] = v
                    charidx = max(char, key=char.get)
                    char_voted.append(self.name_class[charidx])

            else:
                mean_score = []
                for i, array in enumerate(list_array):
                    score = list_array[0][:, 5].mean() * model_weights[i]
                    mean_score.append(score)
                maxidx = np.argmax(mean_score)
                bcs = list_array[maxidx]
                char_voted = [self.name_class[char_id]
                              for char_id in bcs[:, 4]]

        return "".join(char_voted)

    def weighted_voting(self, list_result):
        result = {}
        for i in range(len(list_result)):
            w = self.model_weights[i]
            if list_result[i] not in result:
                result[list_result[i]] = [w]
            else:
                result[list_result[i]].append(w)
        for key in result:
            result[key] = np.sum(result[key])

        return max(result, key=result.get)

    def softvote_words(self, list_array):
        scores = []
        for arr in list_array:
            mean_score = arr[:, 5].mean()
            scores.append(mean_score)

        return np.argmax(scores)

    def cleaning_string(self, my_string):
        my_string = re.sub(r"[^A-Z ]+", '', my_string.upper()
                           )  # change special char to ''
        my_string = " ".join(my_string.split())  # remove white space

        return my_string

    def predict_ensemble(self, img):

        list_result = []
        list_array = []

        for predictor in self.list_predictor:
            final_pred, boxes_class_score = self.predict(
                predictor, img)

            list_result.append(final_pred)
            list_array.append(boxes_class_score)

        # voting from total match
        values, counts = np.unique(list_result, return_counts=True)
        maxidx = np.argmax(counts)
        word_hardvote = values[maxidx]

        # voting from total mean score
        idx = self.softvote_words(list_array)
        word_softvote = list_result[idx]

        # voting with weight
        word_weight_hardvote = self.weighted_voting(list_result)

        # voting char by char
        char_hardvote = self.cleaning_string(
            self.rule_O(self.char_voting(list_array)))
        char_softvote = self.cleaning_string(
            self.rule_O(self.char_voting(list_array, mode='soft')))
        char_weight_hardvote = self.cleaning_string(self.rule_O(self.char_voting(list_array,
                                                                                 weighted=True)))
        char_weight_softvote = self.cleaning_string(self.rule_O(self.char_voting(list_array, mode='soft',
                                                                                 weighted=True)))

        dict_ensemble = dict()
        names = ['hardvote_word', 'hardvote_char', 'softvote_char', 'softvote_word',
                 'weighted_hardvote_word', 'weighted_hardvote_char', 'weighted_softvote_char']
        results = [word_hardvote, char_hardvote, char_softvote, word_softvote,
                   word_weight_hardvote, char_weight_hardvote, char_weight_softvote]
        for nam, res in list(zip(names, results)):
            dict_ensemble[nam] = res

        return dict_ensemble

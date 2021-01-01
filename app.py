from flask import *
from PIL import Image
from face_detector.interface import FaceDetectorInterface
from smile_classifier.interface import is_smiling
from eye_detector.interface import EyeDetectorInterface
from eye_classifier.interface import EyeClassifierInterface

from matplotlib import cm
import cv2
import numpy as np
import sys
import h5py

import base64
import io


app = Flask(__name__)
data_folder = 'model_data'


@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  


@app.route('/model_result', methods = ['POST'])  
def model_result():  
    if request.method == 'POST':  
        f = request.files['file']
        img = np.array(Image.open(f))

        face_detector = FaceDetectorInterface.create(folder='model_data')
        result = face_detector.run(img)

        annotated_image = annotate_img(img, result)
        smile_result, eye_result, pic_result = predict(img, result, annotated_image)

        classification_result = []
        for i in range(len(smile_result)):
            individual_result = "Person {left_count}: " \
                                "Mouth: {smile_res}, " \
                                "Eyes: {eye_res}".format(left_count=i,
                                                            smile_res=smile_result[i],
                                                            eye_res=eye_result[i])
            classification_result.append(individual_result)

        overall_res = "GREAT PHOTO!!" if pic_result else "NOT SO GOOD.."
        res = {"overall picture result": overall_res, "results in detail": classification_result}

        data = io.BytesIO()
        annotated_image = Image.fromarray(annotated_image)
        annotated_image.save(data, "PNG")
        encoded_img_data = base64.b64encode(data.getvalue())
        return render_template("model_result.html", result = json.dumps(res),
                               overall = overall_res,
                               img_data = encoded_img_data.decode('utf-8'))


# return annotated image and list of tuples: (leftmost x-coordinate of bounding box, index of face in person_list)
def annotate_img(image, results):
    image = np.copy(image)
    for ind in range(len(results)):
        result = results[ind]
        start = result['x1'], result['y1']
        end = result['x2'], result['y2']
        image = cv2.rectangle(image, start, end, (0, 255, 0), 2)
        image = cv2.putText(image, str(ind), start, cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2, cv2.LINE_AA)
    return image


# Return smile_classificaiton_result, eye_classification_result, pic_result predicted by the models
# pic_result is a boolean value of if good or not
# Both result lists are in the same order as the face_detector result
def predict(img, res, annotated_image):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = []
    color_faces = []
    for result in res:
        x1 = result['x1']
        x2 = result['x2']
        y1 = result['y1']
        y2 = result['y2']
        
        # extract the actual face image from the box coordinates
        faces.append(cv2.getRectSubPix(gray_img,
                                       (x2 - x1, y2 - y1),
                                       ((x2 + x1) / 2, (y2 + y1) / 2)))
        
        color_faces.append(cv2.getRectSubPix(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB),
                                             (x2 - x1, y2 - y1),
                                             ((x2 + x1) / 2, (y2 + y1) / 2)))
    
    pic_result = True

    smile_out = ['not smiling', 'smiling']
    smile_result = []
    for face in faces:
        smile_res = is_smiling(face, data_folder)
        smile_result.append(smile_out[smile_res])
        pic_result = pic_result and smile_res
    
    eye_detector = EyeDetectorInterface.create(data_folder)
    eye_classifier = EyeClassifierInterface.create(data_folder)

    eye_result = []
    for i, face in enumerate(faces):
        try:
            eyes, radius = eye_detector.detect(face)
            eye1, eye2 = eyes
        except:
            # failed to find an eye from this image, skip to next one
            # doesn't change prediction of image result
            eye_result.append(("Eye 1 Not found", "Eye 2 Not found"))
            continue
        
        diameter = 2 * radius

        # Drawing eyes on image
        topleftx, toplefty = res[i]['x1'], res[i]['y1']
        for eye in [eye1, eye2]:
            start = eye[0] - radius + topleftx, eye[1] - radius + toplefty
            end = eye[0] + radius + topleftx, eye[1] + radius + toplefty
            cv2.rectangle(annotated_image, start, end, (0, 255, 0), 2)

        eye1 = cv2.getRectSubPix(color_faces[i], (diameter, diameter), tuple(eye1))
        eye2 = cv2.getRectSubPix(color_faces[i], (diameter, diameter), tuple(eye2))

        result1 = eye_classifier.run(Image.fromarray(eye1))
        result2 = eye_classifier.run(Image.fromarray(eye2))

        res1 = result1['classes'][result1['label']]
        res2 = result2['classes'][result2['label']]

        eye_result.append((res1, res2))
        pic_result = pic_result and result1['label'] and result2['label']
    
    return smile_result, eye_result, pic_result


if __name__ == '__main__':  
    app.run(debug = True, port = 5000)

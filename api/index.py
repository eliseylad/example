from flask import Flask, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

def process_mockup(object_img_bin, texture_img_bin):
    # Декодируем основной объект (шаблон)
    nparr_obj = np.frombuffer(object_img_bin, np.uint8)
    obj_img = cv2.imdecode(nparr_obj, cv2.IMREAD_UNCHANGED)

    # Декодируем текстуру
    nparr_tex = np.frombuffer(texture_img_bin, np.uint8)
    tex_img = cv2.imdecode(nparr_tex, cv2.IMREAD_COLOR)

    if obj_img is None or tex_img is None:
        return None

    # Логика обработки (твоя функция)
    if len(obj_img.shape) < 3 or obj_img.shape[2] == 3:
        h, w = obj_img.shape[:2]
        alpha = np.full((h, w), 255, dtype=np.uint8)
        obj_color = obj_img
    else:
        b, g, r, alpha = cv2.split(obj_img)
        obj_color = cv2.merge([b, g, r])

    h, w = obj_color.shape[:2]
    bg_h, bg_w = tex_img.shape[:2]
    
    # Ресайз и кроп текстуры под размер объекта
    new_w = int(bg_w * (h / bg_h))
    tex_resized = cv2.resize(tex_img, (new_w, h))[:, (new_w - w) // 2 : (new_w + w) // 2]
    
    # Если текстура шире/узче после ресайза, подгоняем точно в размер объекта
    tex_resized = cv2.resize(tex_resized, (w, h))

    obj_norm = obj_color.astype(float) / 255.0
    tex_norm = tex_resized.astype(float) / 255.0
    blended = (tex_norm * obj_norm) * 255.0
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    b_final, g_final, r_final = cv2.split(blended)
    result = cv2.merge([b_final, g_final, r_final, alpha])

    is_success, buffer = cv2.imencode(".png", result)
    return io.BytesIO(buffer)

@app.route('/api/generate', methods=['POST'])
def generate():
    # Проверяем наличие обоих файлов в запросе
    if 'object' not in request.files or 'texture' not in request.files:
        return {"error": "Нужно отправить два файла: 'object' и 'texture'"}, 400

    obj_file = request.files['object']
    tex_file = request.files['texture']

    result_buffer = process_mockup(obj_file.read(), tex_file.read())
    
    if result_buffer is None:
        return {"error": "Ошибка обработки изображений"}, 400
    
    return send_file(result_buffer, mimetype='image/png', download_name='mockup.png')

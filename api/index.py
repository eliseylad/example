from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

def process_mockup(object_img_bin, texture_img_bin):
    try:
        # Декодируем основной объект (шаблон)
        nparr_obj = np.frombuffer(object_img_bin, np.uint8)
        obj_img = cv2.imdecode(nparr_obj, cv2.IMREAD_UNCHANGED)

        # Декодируем текстуру
        nparr_tex = np.frombuffer(texture_img_bin, np.uint8)
        tex_img = cv2.imdecode(nparr_tex, cv2.IMREAD_COLOR)

        if obj_img is None or tex_img is None:
            return None

        # Проверка альфа-канала у объекта
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
        scale = h / bg_h
        new_w = int(bg_w * scale)
        tex_resized = cv2.resize(tex_img, (new_w, h))
        
        # Центрируем и обрезаем лишнее
        start_x = max(0, (new_w - w) // 2)
        tex_cropped = tex_resized[:, start_x : start_x + w]
        
        # На всякий случай еще раз подгоняем размер (защита от ошибок округления)
        tex_final = cv2.resize(tex_cropped, (w, h))

        # Наложение (Multiply)
        obj_norm = obj_color.astype(float) / 255.0
        tex_norm = tex_final.astype(float) / 255.0
        blended = (tex_norm * obj_norm) * 255.0
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Собираем результат с прозрачностью
        b_f, g_f, r_f = cv2.split(blended)
        result = cv2.merge([b_f, g_f, r_f, alpha])

        # Кодируем в PNG
        is_success, buffer = cv2.imencode(".png", result)
        if not is_success:
            return None
            
        return io.BytesIO(buffer)
    except Exception as e:
        print(f"Error processing: {e}")
        return None

@app.route('/')
def index():
    # Отдаем HTML интерфейс из папки static (она должна быть в корне проекта)
    return send_file('../static/index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    if 'object' not in request.files or 'texture' not in request.files:
        return {"error": "Отправьте оба файла: 'object' и 'texture'"}, 400

    obj_file = request.files['object']
    tex_file = request.files['texture']

    result_io = process_mockup(obj_file.read(), tex_file.read())
    
    if result_io is None:
        return {"error": "Ошибка при обработке изображений. Проверьте формат файлов."}, 400
    
    return send_file(
        result_io, 
        mimetype='image/png', 
        as_attachment=True, 
        download_name='mockup_result.png'
    )

# Для запуска локально (на Vercel не используется)
if __name__ == '__main__':
    app.run(debug=True)

import os
from flask import Flask, render_template, jsonify, request
from mmseg.apis import init_model, inference_model, show_result_pyplot
import jinja2
import torch
import numpy as np
import asyncio
UPLOAD_FOLDER  = os.path.join('static', 'uploads')
from PIL import Image
import time
app = Flask(__name__)
app.config['UPLOAD'] = UPLOAD_FOLDER 
app.jinja_env.globals.update(zip=zip)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]




@app.route('/', methods=['GET', 'POST'])
def main():
    start_time = time.time()
    # if upload file
    if request.method == "POST":
        files = request.files.getlist("files")
        # if upload file success

        if files:
            input_files = save_files(files)
            # init
            areas, scores, predict_files, file_names= [], [], [], []
            tasks = []
            total_score = 0
            for batch_input_file in batch(input_files, 16):
                # input_file = [input_file]
                results = inference_model(model, batch_input_file)
                # tasks.append(save_results(results, batch_input_file))
                save_results(results, batch_input_file)
                for result in results:
                    predict_result = result.pred_sem_seg.data.cpu()
                    area = torch.sum(predict_result) / (predict_result.size()[1] * predict_result.size()[2])
                    area = np.round(area.numpy()*100, 2)
                    if area >= 30:
                        score = 2
                    elif area >= 5:
                        score = 1
                    else:
                        score = 0
                    total_score += score
                    areas.append(area)
                    scores.append(score)

            predict_files += get_predict_files()
            file_names += get_file_name(input_files)
            # asyncio.run(asyncio.wait(tasks))
            end_time = time.time()
            execution_time = end_time - start_time
            print("程式執行時間：", execution_time, "秒")
            return render_template('index.html',
                                   imgs=predict_files,
                                   file_names=file_names,
                                   file_nums=len(input_files),
                                   areas=areas,
                                   scores = scores,
                                   total_score = total_score
                                   )
        # if upload file unsuccess
        else: 
            delete_files_in_folder('static/uploads')
            delete_files_in_folder('static/predict_img')
            return render_template('index.html')
    return render_template('index.html')



def save_results(results, img_list):
    for i, (result, img_path) in enumerate(zip(results, img_list)):
        save_name = os.path.basename(img_path)
        save_dir = 'static/predict_img'
        save_path = os.path.join(save_dir, save_name)
        vis_iamge = show_result_pyplot(model,
                                        img_path,
                                        result,
                                        show = False,
                                        draw_gt = False,
                                        save_dir="",
                                        out_file=save_path)
        
def save_files(files):
    output_files = []
    for file in files:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD'], filename))
        im = Image.open(os.path.join(app.config['UPLOAD'], filename))
        im = im.crop((160, 24, 540, 465))
        im.save(os.path.join(app.config['UPLOAD'], filename))
        file_path = os.path.join(app.config['UPLOAD'], filename)
        output_files.append(file_path)
    return output_files

def get_predict_files():
    predict_folder = os.path.join('static', 'predict_img')
    # read all files in predict folder
    files = os.listdir(predict_folder)
    # get all file path
    files_path = [os.path.join(predict_folder, file) for file in files]
    return files_path

def load_model(config_path, checkpoint_path):
    # Load models into memory
    model = init_model(config_path, checkpoint_path, 'cuda')
    print("model loaded")
    return model

def get_file_name(files):
    import os
    output_files = [os.path.basename(file).split(".")[0] for file in files]
    # get file name
    return output_files

# given a folder, delete all files in it
def delete_files_in_folder(folder):
    files = os.listdir(folder)
    for file in files:
        os.remove(os.path.join(folder, file))


global model
config_path = 'model/mask2former_FocalNet_tiny_50_IM_Aug_focal_decoder.py'
checkpoint_path = 'model/epoch_50.pth'
model = load_model(config_path, checkpoint_path)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5411, debug=True, threaded=True)
    finally:
        delete_files_in_folder('static/uploads')
        delete_files_in_folder('static/predict_img')
    

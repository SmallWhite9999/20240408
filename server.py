from flask import Flask, jsonify
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/get_model_params', methods=['GET'])
def get_model_params():
    try:
        # 从文件中读取模型参数
        with open('hmm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # 将 NumPy 数组转换为 Python 列表
        transmat = model.transmat_.tolist()
        startprob = model.startprob_.tolist()
        means = model.means_.tolist()
        covars = model.covars_.tolist()
        
        # 返回模型参数
        params = {
            'transmat': transmat,
            'startprob': startprob,
            'means': means,
            'covars': covars
        }
        
        # 返回模型参数
        return jsonify(params)
    except FileNotFoundError:
        return 'Model file not found!', 404
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run()

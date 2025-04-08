from flask import Flask, request, jsonify, send_file
import pandas as pd
import tempfile
from flask_cors import CORS
import os
import numpy as np
from predicate import AKIPredictor, generate_charts
import chardet
from io import BytesIO
import zipfile

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def detect_encoding(file_stream):
    rawdata = file_stream.read(1000)
    file_stream.seek(0)
    result = chardet.detect(rawdata)
    return result['encoding']
@app.route('/predict', methods=['POST'])
def unified_predict():
    try:
        lab_columns = [
            'aniongap_min', 'aniongap_max', 'albumin_min', 'albumin_max', 'bands_min', 'bands_max',
            'bicarbonate_min', 'bicarbonate_max', 'bilirubin_min', 'bilirubin_max', 'creatinine_min',
            'creatinine_max', 'chloride_min', 'chloride_max', 'glucose_min', 'glucose_max',
            'hematocrit_min', 'hematocrit_max', 'hemoglobin_min', 'hemoglobin_max', 'lactate_min',
            'lactate_max', 'platelet_min', 'platelet_max', 'potassium_min', 'potassium_max',
            'ptt_min', 'ptt_max', 'inr_min', 'inr_max', 'pt_min', 'pt_max', 'sodium_min',
            'sodium_max', 'bun_min', 'bun_max', 'wbc_min', 'wbc_max', 'gender'
        ]
        
        micro_columns = [
            'spec_itemid', 'org_itemid', 'isolate_num', 'ab_itemid',
            'dilution_text', 'dilution_value', 'urineoutput', 'gender'
        ]
        
        
        if 'file' in request.files:
            file = request.files['file']
            
            # 优先检测文件类型（新增）
            file_header = file.stream.read(4)
            file.stream.seek(0)  # 必须重置指针！
            
            # Excel文件特征检测
            if file_header.startswith(b'PK\x03\x04'):
                return jsonify({
                    'success': False,
                    'message': '检测到Excel文件（非纯CSV格式）',
                    'solution': '请在Excel中使用"文件->另存为->CSV UTF-8"'
                }), 400
                
            app.logger.debug(f"File header: {file.stream.read(20)}")  # 打印文件头
            file.stream.seek(0)
            
            # 详细编码检测
            detected_enc = detect_encoding(file.stream)
            app.logger.info(f"Detailed encoding detection: {chardet.detect(file.stream.read(1000))}")
            file.stream.seek(0)
            # 调试文件信息
            app.logger.info(f"Received file: {file.filename}")
            app.logger.info(f"Content type: {file.content_type}")
            
            # 检测文件编码
            try:
                encoding = detect_encoding(file.stream)
                app.logger.info(f"Detected encoding: {encoding}")
                df = pd.read_csv(file, encoding=encoding, header=0)
                # print(df.head())
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'文件编码检测失败: {str(e)}',
                    'common_encodings': ['utf-8', 'gbk', 'utf-16', 'ascii']
                }), 400
            # 列数验证
            
            if len(df.columns) != 47:
                return jsonify({
                    'success': False,
                    'message': f'列数不正确（需要47列，实际{len(df.columns)}列）',
                    'solution': '请下载最新模板核对列数'
                }), 400
            # print("yes")
            # 列名验证（大小写不敏感）
            # expected_columns = [col.strip().lower() for col in (lab_columns + micro_columns)]
            # received_columns = [col.strip().lower() for col in df.columns]
            # print("yes")
            # if received_columns != expected_columns:
            #     mismatch = [(exp, rec) for exp, rec in zip(expected_columns, received_columns) if exp != rec]
            #     return jsonify({
            #         'success': False,
            #         'message': '列名不匹配',
            #         'mismatch_details': mismatch[:3],  # 显示前三个不匹配项
            #         'expected_example': lab_columns[:3] + micro_columns[:3],
            #         'received_example': list(df.columns)[:6]
            #     }), 400
            # 类型验证
            numeric_cols = [col for col in df.columns if col not in ['gender', 'dilution_text']]
            # print(numeric_cols)
            try:
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': '数值列包含非数字内容',
                    'error_columns': list(df[numeric_cols].columns[df[numeric_cols].isnull().any()])[:3]
                }), 400
            lab_df = df[lab_columns]
            micro_df = df[micro_columns]
            # print(lab_df)
            # print(micro_df)
        else:
            # 表单数据处理
            data = request.get_json()
            
            # 强制对齐实验室数据列
            lab_data = data.get('lab', {})
            lab_df = pd.DataFrame([lab_data], columns=lab_columns)
            lab_df = lab_df.reindex(columns=lab_columns, fill_value=np.nan)
            
            # 强制对齐微生物数据列
            micro_data = data.get('micro', {})
            micro_df = pd.DataFrame([micro_data], columns=micro_columns)
            micro_df = micro_df.reindex(columns=micro_columns, fill_value=np.nan)
            micro_df['gender'] = lab_df['gender'].values
            print(micro_data)
            # 类型转换（处理前端可能传字符串的情况）
            numeric_lab_cols = [col for col in lab_columns if col != 'gender']
            lab_df[numeric_lab_cols] = lab_df[numeric_lab_cols].apply(pd.to_numeric, errors='coerce')
            # 修改过
            numeric_micro_cols = [col for col in micro_columns if col not in ['gender', 'dilution_text']]
            # numeric_micro_cols = ['spec_itemid', 'org_itemid', 'isolate_num', 'dilution_value', 'urineoutput']
            micro_df[numeric_micro_cols] = micro_df[numeric_micro_cols].apply(pd.to_numeric, errors='coerce')

            print("处理后的实验室数据：")
            print(lab_df.head())
            print("\n处理后的微生物数据：")
            print(micro_df.head())

        # 统一预测流程
        # predictor = AKIPredictor()
        print("开始预测...")
        # 添加空值检查
        if lab_df.isnull().values.any():
            print("警告：实验室数据中存在空值")
        if micro_df.isnull().values.any():
            print("警告：微生物数据中存在空值")
            
        generate_charts(micro_df, lab_df)
        
        return jsonify({
            'success': True,
            'charts': [
                "gauge.html",   
                "lab_boxplot.html",
                "lab_correlation.html", 
                "lab_pairplot.html",
                "lab_parallel.html",
                "lab_violin.html",
                "prob_chart.html",
                "prob_histogram.html",
                "radar.html",
                "risk_dist.html",
                "sunburst.html"
            ]
        })
    
    except pd.errors.ParserError as e:
        return jsonify({
            'success': False,
            'message': 'CSV解析失败',
            'common_causes': [
                '文件包含合并单元格',
                '使用了错误的分隔符',
                '存在未闭合的引号',
                '文件包含二进制数据'
            ]
        }), 400
    except KeyError as e:
        return jsonify({'success': False, 'message': f'缺少必要字段: {str(e)}'}), 400
    except pd.errors.ParserError as e:
        return jsonify({'success': False, 'message': f'CSV解析错误: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'success': False, 'message': f'数据类型错误: {str(e)}'}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'服务器内部错误: {str(e)}'}), 500

@app.route('/download_template')
def download_template():
    # 生成符合要求的空模板
    expected_columns = lab_columns + micro_columns
    df = pd.DataFrame(columns=expected_columns)
    
    # 添加示例数据
    example_data = {
        'aniongap_min': [12, 14],
        'gender': ['男', '女']
    }
    df = df.append(pd.DataFrame(example_data))
    
    # 保存到临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    df.to_csv(temp_file.name, index=False, encoding='utf-8-sig')
    
    return send_file(
        temp_file.name,
        mimetype='text/csv',
        as_attachment=True,
        attachment_filename='predict_template.csv'
    )

@app.route('/get_chart/<filename>')
def get_chart(filename):
    # 修正图表路径
    chart_path = os.path.join('output', filename)
    if not os.path.exists(chart_path):
        return "Chart not found", 404
    return send_file(chart_path, mimetype='text/html')


@app.route('/download_model', methods=['GET'])
def download_model():
    try:
        # 修正模型目录路径
        model_dir = os.path.join(app.root_path, 'public', 'models')
        
        # 调试日志增强
        app.logger.info(f"应用根目录：{app.root_path}")
        app.logger.info(f"最终模型目录：{model_dir}")
        app.logger.info(f"目录内容：{os.listdir(model_dir)}")
        
        if not os.path.exists(model_dir):
            return jsonify({
                'success': False,
                'message': f'模型目录不存在，当前路径：{model_dir}'
            }), 404

        # 创建内存ZIP时添加路径校验
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(model_dir):
                if not files:  # 空目录检查
                    app.logger.warning(f'空目录：{root}')
                    continue
                    
                for file in files:
                    file_path = os.path.join(root, file)
                    if not os.path.isfile(file_path):  # 非文件检查
                        continue
                        
                    arcname = os.path.relpath(file_path, model_dir)
                    zf.write(file_path, arcname=arcname)
                    app.logger.debug(f'已添加文件：{arcname}')

        memory_file.seek(0)
        
        # 添加响应头验证
        headers = {
            'Content-Type': 'application/zip',
            'Content-Disposition': 'attachment; filename="model_package.zip"'
        }
        return send_file(
            memory_file,
            as_attachment=True,
            download_name='model_package.zip',
            mimetype='application/zip'
        )

    except Exception as e:
        app.logger.error(f'打包失败：{str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'message': f'服务器处理错误：{str(e)}',
            'expected_path': os.path.join(app.root_path, 'public', 'download_model')
        }), 500




if __name__ == '__main__':
    # 创建输出目录
    if not os.path.exists('output'):
        os.makedirs('output')
    app.run(host='0.0.0.0', port=5000, debug=True)